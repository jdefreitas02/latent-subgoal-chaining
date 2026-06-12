from typing import Any

import copy
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, ActorVectorField, GCActor, GCValue, Identity, LengthNormalize


class HIQLACFQLAgent(flax.struct.PyTreeNode):
    """HIQL hierarchy + ACFQL flow-matching low-level actor with action chunking.

    The high level and value/representation are identical to HIQL. The low-level
    actor is replaced by a flow-matching vector field that natively emits a
    H_c-step action chunk (flattened to action_dim * action_chunk_len), trained
    with advantage-weighted flow matching (the flow analogue of HIQL's AWR
    Gaussian low actor). This removes the 5->25 decoder bottleneck: the chunk the
    LL emits is exactly what a downstream world model would consume.

    Goal-conditioning of the flow actor is done by concatenating the subgoal
    representation phi([s; w]) to the state and passing it as the actor's
    observation input.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()
    # Stage B (WGSP): frozen LeJEPA world model used only to RANK the LL's own
    # chunk samples (never a value-learning target). wm_model is a static Flax
    # module; wm_params is a constant pytree carried along (no grad). Both None
    # for Stage A1/A2 (wgsp_coef=0 -> the WGSP branch is not traced).
    wm_model: Any = nonpytree_field(default=None)
    wm_params: Any = None

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def value_loss(self, batch, grad_params):
        """Compute the IVL value loss (identical to HIQL)."""
        (next_v1_t, next_v2_t) = self.network.select('target_value')(batch['next_observations'], batch['value_goals'])
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v_t

        (v1_t, v2_t) = self.network.select('target_value')(batch['observations'], batch['value_goals'])
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v1_t
        q2 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v2_t
        (v1, v2) = self.network.select('value')(batch['observations'], batch['value_goals'], params=grad_params)
        v = (v1 + v2) / 2

        value_loss1 = self.expectile_loss(adv, q1 - v1, self.config['expectile']).mean()
        value_loss2 = self.expectile_loss(adv, q2 - v2, self.config['expectile']).mean()
        value_loss = value_loss1 + value_loss2

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def low_actor_loss(self, batch, grad_params, rng):
        """Advantage-weighted flow-matching low-level actor loss.

        adv  = V(s_{t+H_c}, g_low) - V(s_t, g_low)        (H_c-step advantage)
        w    = min(exp(low_alpha * adv), 100)
        flow BC on the H_c-step chunk conditioned on [s_t, phi([s_t; g_low])],
        each sample weighted by w. This is the flow analogue of HIQL's AWR
        Gaussian low actor; targets are data chunks (in-distribution -> bounded).
        """
        # H_c-step advantage toward the subgoal.
        v1, v2 = self.network.select('value')(batch['observations'], batch['low_actor_goals'])
        nv1, nv2 = self.network.select('value')(batch['low_chunk_next_observations'], batch['low_actor_goals'])
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v

        exp_a = jnp.exp(adv * self.config['low_alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        # Subgoal representation phi([s; g_low]); condition the flow on [s, rep].
        goal_reps = self.network.select('goal_rep')(
            jnp.concatenate([batch['observations'], batch['low_actor_goals']], axis=-1),
            params=grad_params,
        )
        if not self.config['low_actor_rep_grad']:
            goal_reps = jax.lax.stop_gradient(goal_reps)
        obs_cond = jnp.concatenate([batch['observations'], goal_reps], axis=-1)

        chunk = batch['action_chunks']  # (B, action_dim * action_chunk_len)
        batch_size, chunk_dim = chunk.shape

        # Flow-matching BC target.
        rng, x_rng, t_rng = jax.random.split(rng, 3)
        x_0 = jax.random.normal(x_rng, (batch_size, chunk_dim))
        x_1 = chunk
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('low_actor_bc_flow')(obs_cond, x_t, t, is_encoded=True, params=grad_params)

        # Per-sample flow loss. ll_awr=True -> AWR-weighted flow matching (HIQL
        # analogue); ll_awr=False -> uniform-weight plain BC flow (the literal
        # q-chunking / ACFQL recipe, where 'goodness' comes from inference-time
        # selection rather than a train-time advantage weight).
        flow_loss_per = jnp.mean((pred - vel) ** 2, axis=-1)  # (B,)
        weight = exp_a if self.config['ll_awr'] else jnp.ones_like(exp_a)
        actor_loss = (weight * flow_loss_per).mean()

        return actor_loss, {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'flow_bc_loss': flow_loss_per.mean(),
            'exp_a_mean': exp_a.mean(),
            'exp_a_max': exp_a.max(),
        }

    def wgsp_low_actor_loss(self, batch, grad_params, rng):
        """Stage B: world-model-grounded improvement term for the LL flow actor.

        From REAL start latents z_t (tau=0; no imagined-state contamination), draw
        M chunk candidates from the CURRENT LL (stop-grad sampling), roll each one
        step through the FROZEN WM to z'_m, score J_m = V(z'_m, g_low) (V is a
        constant here -- not differentiated, so the WM/value only RANK), form a
        group-relative advantage A_m = J_m - mean_m J_m, and regress the LL flow
        (with grad) toward its own high-advantage samples, weighted by
        w_m = softmax(wgsp_alpha * A_m). This is advantage-weighted flow matching
        (RWR/GRPO-on-flow) toward WM-preferred chunks. The data-chunk BC term in
        low_actor_loss anchors the policy to the data manifold; this term nudges it
        toward chunks the WM predicts reach higher value. The decoder is gone: the
        25-D chunk the LL emits IS the WM's action input.
        """
        wb = self.config['wgsp_batch_size']
        M = self.config['wgsp_num_samples']
        chunk_dim = self.config['action_dim'] * self.config['action_chunk_len']

        z_t = batch['observations'][:wb]            # (wb, d)  real latents
        g_low = batch['low_actor_goals'][:wb]       # (wb, d)
        wb = z_t.shape[0]

        # Subgoal-rep conditioning (stop-grad rep; the WGSP signal trains the flow,
        # not the representation).
        rep = self.network.select('goal_rep')(jnp.concatenate([z_t, g_low], axis=-1))
        rep = jax.lax.stop_gradient(rep)
        obs_cond = jnp.concatenate([z_t, rep], axis=-1)  # (wb, d+rep)

        # --- Sample M chunks from the current LL (no grad through sampling). ---
        def _sample(key):
            noise = jax.random.normal(key, (wb, chunk_dim))
            ch = self._flow_integrate('low_actor_bc_flow', obs_cond, noise)
            return jnp.clip(ch, -1.0, 1.0)
        keys = jax.random.split(rng, M)
        chunks = jax.vmap(_sample)(keys)                 # (M, wb, chunk_dim)
        chunks = jax.lax.stop_gradient(chunks)

        # --- Optional FMQ refinement: one normalized-grad step ascending
        #     V(WM(z, chunk), g_low), backprop'd through the differentiable WM.
        #     Pushes candidates out of the (possibly narrow) proposal support. ---
        if self.config['wgsp_fmq_eta'] > 0.0:
            g_b = jnp.broadcast_to(g_low[None], (M, wb, g_low.shape[-1]))

            def _fmq_score(ch):
                z1 = jax.vmap(lambda c: self.wm_model.apply(
                    self.wm_params, z_t[:, None, :], c[:, None, :])[:, -1, :])(ch)
                v1, v2 = self.network.select('value')(
                    z1.reshape(M * wb, -1), g_b.reshape(M * wb, -1))
                return ((v1 + v2) / 2.0).sum()

            grads = jax.grad(_fmq_score)(chunks)
            grads = grads / (jnp.linalg.norm(grads, axis=-1, keepdims=True) + 1e-8)
            chunks = jnp.clip(chunks + self.config['wgsp_fmq_eta'] * grads, -1.0, 1.0)
            chunks = jax.lax.stop_gradient(chunks)

        # --- Roll each candidate one WM step; score V at the endpoint. ---
        def _roll(ch):
            z1 = self.wm_model.apply(self.wm_params, z_t[:, None, :], ch[:, None, :])[:, -1, :]
            return z1
        z1 = jax.vmap(_roll)(chunks)                     # (M, wb, d)
        g_tiled = jnp.broadcast_to(g_low[None], (M, wb, g_low.shape[-1]))
        v1, v2 = self.network.select('value')(
            z1.reshape(M * wb, -1), g_tiled.reshape(M * wb, -1))
        J = ((v1 + v2) / 2.0).reshape(M, wb)             # constant (no grad_params)
        J = jax.lax.stop_gradient(J)

        # Group-relative advantage + softmax weights across the M candidates.
        adv = J - J.mean(axis=0, keepdims=True)          # (M, wb)
        logits = self.config['wgsp_alpha'] * adv
        w = jax.nn.softmax(logits, axis=0)               # (M, wb), sums to 1 over M

        # --- Advantage-weighted flow BC toward the sampled chunks (with grad). ---
        rng, x_rng, t_rng = jax.random.split(rng, 3)
        x_0 = jax.random.normal(x_rng, (M, wb, chunk_dim))
        x_1 = chunks
        t = jax.random.uniform(t_rng, (M, wb, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0
        obs_cond_tiled = jnp.broadcast_to(obs_cond[None], (M, wb, obs_cond.shape[-1]))
        pred = self.network.select('low_actor_bc_flow')(
            obs_cond_tiled.reshape(M * wb, -1),
            x_t.reshape(M * wb, -1),
            t.reshape(M * wb, 1),
            is_encoded=True, params=grad_params,
        ).reshape(M, wb, chunk_dim)
        flow_loss_per = jnp.mean((pred - vel) ** 2, axis=-1)  # (M, wb)
        wgsp_loss = (w * flow_loss_per).sum(axis=0).mean()

        return wgsp_loss, {
            'wgsp_loss': wgsp_loss,
            'wgsp_J_mean': J.mean(),
            'wgsp_adv_std': adv.std(),
            'wgsp_w_max': w.max(),
        }

    def high_actor_loss(self, batch, grad_params, rng):
        """Compute the high-level actor loss.

        Two forms (config['high_actor_type']):
          'gaussian' — HIQL's AWR Gaussian over the subgoal rep (default ablation).
          'flow'     — AWR-weighted flow matching to the subgoal rep (the flow
                       analogue; diversity is set by the noise, not by ||mu||, so
                       it does not self-throttle as the HL converges — the key
                       property needed for WGSP multi-candidate sampling later).
        """
        v1, v2 = self.network.select('value')(batch['observations'], batch['high_actor_goals'])
        nv1, nv2 = self.network.select('value')(batch['high_actor_targets'], batch['high_actor_goals'])
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v

        exp_a = jnp.exp(adv * self.config['high_alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        # Subgoal-rep target (length-normalized); constant w.r.t. grad (no params=grad_params).
        target = self.network.select('goal_rep')(
            jnp.concatenate([batch['observations'], batch['high_actor_targets']], axis=-1)
        )

        if self.config['high_actor_type'] == 'flow':
            obs_cond = jnp.concatenate([batch['observations'], batch['high_actor_goals']], axis=-1)
            rep_dim = target.shape[-1]
            rng, x_rng, t_rng = jax.random.split(rng, 3)
            x_0 = jax.random.normal(x_rng, (target.shape[0], rep_dim))
            x_1 = target
            t = jax.random.uniform(t_rng, (target.shape[0], 1))
            x_t = (1 - t) * x_0 + t * x_1
            vel = x_1 - x_0
            pred = self.network.select('high_actor_flow')(obs_cond, x_t, t, is_encoded=True, params=grad_params)
            flow_loss_per = jnp.mean((pred - vel) ** 2, axis=-1)
            actor_loss = (exp_a * flow_loss_per).mean()
            return actor_loss, {
                'actor_loss': actor_loss,
                'adv': adv.mean(),
                'flow_bc_loss': flow_loss_per.mean(),
                'exp_a_mean': exp_a.mean(),
            }

        dist = self.network.select('high_actor')(batch['observations'], batch['high_actor_goals'], params=grad_params)
        log_prob = dist.log_prob(target)
        actor_loss = -(exp_a * log_prob).mean()
        return actor_loss, {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - target) ** 2),
            'std': jnp.mean(dist.scale_diag),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng
        rng, low_rng, high_rng, wgsp_rng = jax.random.split(rng, 4)

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        low_actor_loss, low_actor_info = self.low_actor_loss(batch, grad_params, low_rng)
        for k, v in low_actor_info.items():
            info[f'low_actor/{k}'] = v

        high_actor_loss, high_actor_info = self.high_actor_loss(batch, grad_params, high_rng)
        for k, v in high_actor_info.items():
            info[f'high_actor/{k}'] = v

        loss = value_loss + low_actor_loss + high_actor_loss

        # Stage B: WM-grounded LL improvement term (static branch; not traced for A1/A2).
        if self.config['wgsp_coef'] > 0.0:
            wgsp_loss, wgsp_info = self.wgsp_low_actor_loss(batch, grad_params, wgsp_rng)
            for k, v in wgsp_info.items():
                info[f'wgsp/{k}'] = v
            loss = loss + self.config['wgsp_coef'] * wgsp_loss

        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'value')

        return self.replace(network=new_network, rng=new_rng), info

    def _flow_integrate(self, module_name, obs_cond, noise):
        """Euler-integrate a flow vector field from noise to a sample."""
        x = noise
        for i in range(self.config['flow_steps']):
            t = jnp.full((*obs_cond.shape[:-1], 1), i / self.config['flow_steps'])
            vel = self.network.select(module_name)(obs_cond, x, t, is_encoded=True)
            x = x + vel / self.config['flow_steps']
        return x

    @jax.jit
    def sample_high_rep(self, observations, goals, seed=None, temperature=1.0):
        """Sample a subgoal representation from the high-level actor (gaussian or flow)."""
        if self.config['high_actor_type'] == 'flow':
            obs_cond = jnp.concatenate([observations, goals], axis=-1)
            noise = jax.random.normal(seed, (*observations.shape[:-1], self.config['rep_dim']))
            goal_reps = self._flow_integrate('high_actor_flow', obs_cond, noise)
        else:
            high_dist = self.network.select('high_actor')(observations, goals, temperature=temperature)
            goal_reps = high_dist.sample(seed=seed)
        goal_reps = goal_reps / jnp.linalg.norm(goal_reps, axis=-1, keepdims=True) * jnp.sqrt(goal_reps.shape[-1])
        return goal_reps

    @jax.jit
    def sample_low_chunk(self, observations, goal_reps, seed=None, temperature=1.0):
        """Integrate the BC flow to produce a H_c-step action chunk.

        Returns the flattened chunk of shape (..., action_dim * action_chunk_len),
        clipped to [-1, 1]. The eval loop reshapes it to (action_chunk_len, action_dim).
        """
        chunk_dim = self.config['action_dim'] * self.config['action_chunk_len']
        obs_cond = jnp.concatenate([observations, goal_reps], axis=-1)
        noises = jax.random.normal(seed, (*observations.shape[:-1], chunk_dim))
        actions = self._flow_integrate('low_actor_bc_flow', obs_cond, noises)
        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
        wm_model=None,
        wm_params=None,
    ):
        """Create a new agent.

        wm_model / wm_params (optional): a frozen LeJEPA JAX world model
        (LeJEPAJaxForward + its params) used only by the Stage B WGSP term to rank
        the LL's own chunk samples. Required iff config['wgsp_coef'] > 0.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        action_dim = ex_actions.shape[-1]
        chunk_dim = action_dim * config['action_chunk_len']

        # Subgoal representation phi([s; g]) -> length-normalized rep.
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            goal_rep_seq = [encoder_module()]
        else:
            goal_rep_seq = []
        goal_rep_seq.append(
            MLP(
                hidden_dims=(*config['value_hidden_dims'], config['rep_dim']),
                activate_final=False,
                layer_norm=config['layer_norm'],
            )
        )
        goal_rep_seq.append(LengthNormalize())
        goal_rep_def = nn.Sequential(goal_rep_seq)

        if config['encoder'] is not None:
            value_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            target_value_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            high_actor_encoder_def = GCEncoder(concat_encoder=encoder_module())
            # Flow actor encoder applied to the [s, rep] conditioning vector. For
            # pixel obs, the rep is concatenated AFTER encoding, so we keep the
            # flow encoder None and pre-encode upstream (handled in A2).
            low_flow_encoder_def = None
        else:
            value_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            target_value_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            high_actor_encoder_def = None
            low_flow_encoder_def = None

        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=value_encoder_def,
        )
        target_value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=target_value_encoder_def,
        )

        # Low-level flow actor: emits a flattened H_c-step chunk.
        low_actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=chunk_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=low_flow_encoder_def,
            use_fourier_features=config['use_fourier_features'],
            fourier_feature_dim=config['fourier_feature_dim'],
        )

        # Example inputs for init.
        ex_rep = jnp.zeros((*ex_observations.shape[:-1], config['rep_dim']), dtype=jnp.float32)
        ex_obs_cond = jnp.concatenate([ex_observations, ex_rep], axis=-1)
        ex_chunk = jnp.zeros((*ex_observations.shape[:-1], chunk_dim), dtype=jnp.float32)
        ex_times = jnp.zeros((*ex_observations.shape[:-1], 1), dtype=jnp.float32)
        ex_high_obs_cond = jnp.concatenate([ex_observations, ex_goals], axis=-1)

        network_info = dict(
            goal_rep=(goal_rep_def, (jnp.concatenate([ex_observations, ex_goals], axis=-1))),
            value=(value_def, (ex_observations, ex_goals)),
            target_value=(target_value_def, (ex_observations, ex_goals)),
            low_actor_bc_flow=(low_actor_bc_flow_def, (ex_obs_cond, ex_chunk, ex_times)),
        )

        # High-level actor: gaussian (HIQL, default ablation) or flow (diversity-safe).
        if config['high_actor_type'] == 'flow':
            high_actor_flow_def = ActorVectorField(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=config['rep_dim'],
                layer_norm=config['actor_layer_norm'],
                encoder=None,  # state obs; visual A2 pre-encodes upstream
                use_fourier_features=config['use_fourier_features'],
                fourier_feature_dim=config['fourier_feature_dim'],
            )
            network_info['high_actor_flow'] = (high_actor_flow_def, (ex_high_obs_cond, ex_rep, ex_times))
        else:
            high_actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=config['rep_dim'],
                state_dependent_std=False,
                const_std=config['const_std'],
                gc_encoder=high_actor_encoder_def,
            )
            network_info['high_actor'] = (high_actor_def, (ex_observations, ex_goals))
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_value'] = params['modules_value']

        config['action_dim'] = action_dim
        if config['wgsp_coef'] > 0.0:
            assert wm_model is not None and wm_params is not None, \
                'wgsp_coef>0 requires a frozen WM (wm_model, wm_params)'
        return cls(rng, network=network, config=flax.core.FrozenDict(**config),
                   wm_model=wm_model, wm_params=wm_params)


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='hiql_acfql',
            lr=3e-4,
            batch_size=1024,
            actor_hidden_dims=(512, 512, 512),
            value_hidden_dims=(512, 512, 512),
            layer_norm=True,
            actor_layer_norm=False,
            discount=0.99,
            tau=0.005,
            expectile=0.7,
            low_alpha=3.0,  # Low-level AWR temperature.
            high_alpha=3.0,  # High-level AWR temperature.
            high_actor_type='gaussian',  # 'gaussian' (HIQL, default ablation) or 'flow'.
            ll_awr=True,  # True: AWR-weighted flow LL. False: plain BC flow (q-chunking-style).
            # Stage B (WGSP) WM-grounded LL improvement term. wgsp_coef=0 -> off
            # (pure A1/A2; WM not traced). >0 -> add the model-grounded term.
            wgsp_coef=0.0,
            wgsp_num_samples=4,    # M chunk candidates per state, ranked by the WM.
            wgsp_batch_size=256,   # sub-batch used for the (costlier) WGSP term.
            wgsp_alpha=3.0,        # softmax temperature on the group-relative WM advantage.
            # FMQ refinement: one normalized-gradient step on each sampled chunk to
            # ascend V(WM(z,chunk), g) before ranking. 0 -> off (pure best-of-N).
            # Lets candidates leave the proposal's support (best-of-N cannot).
            wgsp_fmq_eta=0.0,
            subgoal_steps=10,  # HL replan horizon (env steps).
            action_chunk_len=5,  # LL chunk length (env steps); chunk_dim = action_dim * this.
            flow_steps=10,  # Euler integration steps for the BC flow.
            rep_dim=10,
            low_actor_rep_grad=False,  # Whether LL flow grads flow into goal_rep (True for pixels).
            const_std=True,
            discrete=False,
            encoder=ml_collections.config_dict.placeholder(str),
            use_fourier_features=False,
            fourier_feature_dim=64,
            # Hierarchical chunked eval driver flag (read by main.py).
            hierarchical_chunked=True,
            action_dim=ml_collections.config_dict.placeholder(int),  # set automatically
            # Dataset hyperparameters.
            dataset_class='HGCDataset',
            value_p_curgoal=0.2,
            value_p_trajgoal=0.5,
            value_p_randomgoal=0.3,
            value_geom_sample=True,
            actor_p_curgoal=0.0,
            actor_p_trajgoal=1.0,
            actor_p_randomgoal=0.0,
            actor_geom_sample=False,
            gc_negative=True,
            p_aug=0.0,
            frame_stack=ml_collections.config_dict.placeholder(int),
        )
    )
    return config
