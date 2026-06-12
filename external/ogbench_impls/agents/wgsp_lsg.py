"""WGSP-LSG: hierarchical World-model-Grounded Subgoal Planning, latent OR rep subgoals.

True subgoal grounding (vs the chunk-reranking that 'WGSP' had drifted into),
built to reuse the FMQ-MPC machinery of the working online-in-WM method
(sec:mpc-online-wm). The high level proposes a subgoal via a flow policy; the low
level is conditioned on that subgoal and trained to REACH it; a frozen value
grounds both at distillation time.

subgoal_space ABLATION:
  'latent' -- HL emits a 192-D LeJEPA latent subgoal w (flow; a diagonal Gaussian
     over 192-D could not stay on the manifold). Value V(z,g) is plain (no rep
     bottleneck). LL reachability can be scored by latent distance ||z'-w|| (the
     WM's native CEM metric -- impossible with a rep) or by value V(z',w).
  'rep' -- the original HIQL-style 10-D subgoal rep phi([s;g]) (length-normalized),
     learned as the value's bottleneck V(z, phi([z;g])). LL conditions on the rep;
     reachability is scored by value (a rep cannot be distance-compared to a WM
     latent). This is the 'original 10-dim, no latent metric' baseline.

Two-phase training:
  Phase 1 (total_loss): hierarchical pretrain -- IQL value, AWR flow-BC HL toward
    data subgoals, flow-BC LL conditioned on the subgoal.
  Phase 2 (trainer-driven, V FROZEN): WGSP-MPC distillation -- LL distilled toward
    FMQ-MPC chunks that reach the subgoal; HL AWR-grounded by the value the LL
    achieves when aiming for each candidate subgoal.
"""
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, GCValue, MLP, LengthNormalize


class WGSPLSGAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    config: Any = nonpytree_field()
    wm_model: Any = nonpytree_field(default=None)   # frozen LeJEPA WM (Phase-2 ranker only)
    wm_params: Any = None

    # ------------------------------------------------------------------ #
    # Subgoal-space helpers (the only thing that branches latent vs rep).
    # ------------------------------------------------------------------ #
    def _value(self, z, g, params=None, target=False):
        """Goal-conditioned value, returning the ensemble tuple (v1, v2).
        rep mode bottlenecks the goal through phi: V(z, phi([z;g]))."""
        vname = 'target_value' if target else 'value'
        if self.config['subgoal_space'] == 'rep':
            rname = 'target_goal_rep' if target else 'goal_rep'
            rep = self.network.select(rname)(jnp.concatenate([z, g], axis=-1), params=params)
            return self.network.select(vname)(z, rep, params=params)
        return self.network.select(vname)(z, g, params=params)

    def _subgoal_from_latent(self, z, w_latent):
        """Map a latent waypoint w to the subgoal the LL/HL use: itself (latent mode)
        or its (stop-grad) rep phi([z;w]) (rep mode)."""
        if self.config['subgoal_space'] == 'rep':
            return jax.lax.stop_gradient(
                self.network.select('goal_rep')(jnp.concatenate([z, w_latent], axis=-1)))
        return w_latent

    # ------------------------------------------------------------------ #
    # Phase 1 losses
    # ------------------------------------------------------------------ #
    @staticmethod
    def expectile_loss(adv, diff, expectile):
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff ** 2)

    def value_loss(self, batch, grad_params):
        next_v1_t, next_v2_t = self._value(batch['next_observations'], batch['value_goals'], target=True)
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v_t

        v1_t, v2_t = self._value(batch['observations'], batch['value_goals'], target=True)
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v1_t
        q2 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v2_t
        v1, v2 = self._value(batch['observations'], batch['value_goals'], params=grad_params)
        v = (v1 + v2) / 2

        value_loss1 = self.expectile_loss(adv, q1 - v1, self.config['expectile']).mean()
        value_loss2 = self.expectile_loss(adv, q2 - v2, self.config['expectile']).mean()
        value_loss = value_loss1 + value_loss2
        return value_loss, {'value_loss': value_loss, 'v_mean': v.mean(),
                            'v_max': v.max(), 'v_min': v.min()}

    def _flow_bc(self, module_name, obs_cond, x_1, rng, weight, grad_params):
        b, dim = x_1.shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)
        x_0 = jax.random.normal(x_rng, (b, dim))
        t = jax.random.uniform(t_rng, (b, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0
        pred = self.network.select(module_name)(obs_cond, x_t, t, is_encoded=True, params=grad_params)
        per = jnp.mean((pred - vel) ** 2, axis=-1)
        return (weight * per).mean(), per.mean()

    def _flow_bc_per_sample(self, module_name, obs_cond, x_1, rng, params):
        """Per-sample flow BC losses, shape (B,). Used by DPO for per-element logit computation."""
        b, dim = x_1.shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)
        x_0 = jax.random.normal(x_rng, (b, dim))
        t = jax.random.uniform(t_rng, (b, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0
        pred = self.network.select(module_name)(obs_cond, x_t, t, is_encoded=True, params=params)
        return jnp.mean((pred - vel) ** 2, axis=-1)  # (B,)

    def high_actor_loss(self, batch, grad_params, rng):
        z, g = batch['observations'], batch['high_actor_goals']
        w_latent = batch['high_actor_targets']
        target = self._subgoal_from_latent(z, w_latent)         # latent w, or rep phi([z;w])

        v1, v2 = self._value(z, g)
        nv1, nv2 = self._value(w_latent, g)
        adv = (nv1 + nv2) / 2 - (v1 + v2) / 2
        weight = jnp.minimum(jnp.exp(adv * self.config['high_alpha']), 100.0)

        obs_cond = jnp.concatenate([z, g], axis=-1)
        loss, bc = self._flow_bc('high_actor_flow', obs_cond, target, rng, weight, grad_params)
        return loss, {'high_actor_loss': loss, 'high_flow_bc': bc,
                      'high_adv': adv.mean(), 'high_w_max': weight.max()}

    def low_actor_loss(self, batch, grad_params, rng):
        z = batch['observations']
        w_latent = batch['high_actor_targets']
        sg = self._subgoal_from_latent(z, w_latent)             # LL conditioning subgoal
        chunk = batch['action_chunks']

        if self.config['ll_awr']:
            v1, v2 = self._value(z, w_latent)
            nv1, nv2 = self._value(batch['low_chunk_next_observations'], w_latent)
            adv = (nv1 + nv2) / 2 - (v1 + v2) / 2
            weight = jnp.minimum(jnp.exp(adv * self.config['low_alpha']), 100.0)
        else:
            weight = jnp.ones((z.shape[0],))

        obs_cond = jnp.concatenate([z, sg], axis=-1)
        loss, bc = self._flow_bc('low_actor_bc_flow', obs_cond, chunk, rng, weight, grad_params)
        return loss, {'low_actor_loss': loss, 'low_flow_bc': bc}

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        rng, low_rng, high_rng = jax.random.split(rng, 3)
        value_loss, vi = self.value_loss(batch, grad_params)
        for k, v in vi.items():
            info[f'value/{k}'] = v
        low_loss, li = self.low_actor_loss(batch, grad_params, low_rng)
        for k, v in li.items():
            info[f'low_actor/{k}'] = v
        high_loss, hi = self.high_actor_loss(batch, grad_params, high_rng)
        for k, v in hi.items():
            info[f'high_actor/{k}'] = v
        return value_loss + low_loss + high_loss, info

    def target_update(self, network, module_name):
        new_tp = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'])
        network.params[f'modules_target_{module_name}'] = new_tp

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'value')
        if self.config['subgoal_space'] == 'rep':
            self.target_update(new_network, 'goal_rep')
        return self.replace(network=new_network, rng=new_rng), info

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #
    def _flow_integrate(self, module_name, obs_cond, noise):
        x = noise
        for i in range(self.config['flow_steps']):
            t = jnp.full((*obs_cond.shape[:-1], 1), i / self.config['flow_steps'])
            x = x + self.network.select(module_name)(obs_cond, x, t, is_encoded=True) / self.config['flow_steps']
        return x

    @jax.jit
    def sample_high_subgoal(self, observations, goals, seed=None):
        """Integrate the HL flow to a subgoal (192-D latent, or 10-D length-normalized rep)."""
        obs_cond = jnp.concatenate([observations, goals], axis=-1)
        noise = jax.random.normal(seed, (*observations.shape[:-1], self.config['subgoal_dim']))
        sg = self._flow_integrate('high_actor_flow', obs_cond, noise)
        if self.config['subgoal_space'] == 'rep':
            sg = sg / (jnp.linalg.norm(sg, axis=-1, keepdims=True) + 1e-8) * jnp.sqrt(sg.shape[-1])
        return sg

    @jax.jit
    def sample_low_chunk(self, observations, subgoals, seed=None):
        """LL chunk conditioned on the subgoal (already in subgoal space: latent or rep)."""
        chunk_dim = self.config['action_dim'] * self.config['action_chunk_len']
        obs_cond = jnp.concatenate([observations, subgoals], axis=-1)
        noise = jax.random.normal(seed, (*observations.shape[:-1], chunk_dim))
        return jnp.clip(self._flow_integrate('low_actor_bc_flow', obs_cond, noise), -1, 1)

    # ------------------------------------------------------------------ #
    # Phase 2: WGSP-MPC distillation primitives (V frozen).
    # ------------------------------------------------------------------ #
    def _reach_score(self, z1, w_latent):
        """Reachability of latent waypoint w from rolled latent z1. 'distance' uses
        the WM-native -||z1 - w|| (latent mode only; a rep cannot express it);
        'value' uses V(z1, w)."""
        if self.config['ll_reach_metric'] == 'distance':
            return -jnp.linalg.norm(z1 - w_latent, axis=-1)
        v1, v2 = self._value(z1, w_latent)
        return (v1 + v2) / 2.0

    @jax.jit
    def fmq_mpc_low(self, z, w_latent, seed):
        """Select the chunk that best REACHES the latent waypoint w. The LL is
        conditioned on the subgoal (rep phi([z;w]) in rep mode, w itself in latent
        mode); scoring uses the latent w (distance or value). One WM step,
        in-distribution proposals -> reliable regime."""
        N = self.config['wgsp_num_samples']
        B = z.shape[0]
        cd = self.config['action_dim'] * self.config['action_chunk_len']
        z_rep = jnp.repeat(z, N, axis=0)
        w_rep = jnp.repeat(w_latent, N, axis=0)
        sg = self._subgoal_from_latent(z_rep, w_rep)            # LL conditioning subgoal
        chunks = self.sample_low_chunk(z_rep, sg, seed=seed)
        if self.config['wgsp_fmq_eta'] > 0.0:
            def _score(ch):
                z1 = self.wm_model.apply(self.wm_params, z_rep[:, None, :], ch[:, None, :])[:, -1, :]
                return self._reach_score(z1, w_rep).sum()
            grd = jax.grad(_score)(chunks)
            grd = grd / (jnp.linalg.norm(grd, axis=-1, keepdims=True) + 1e-8)
            chunks = jnp.clip(chunks + self.config['wgsp_fmq_eta'] * grd, -1.0, 1.0)
        z1 = self.wm_model.apply(self.wm_params, z_rep[:, None, :], chunks[:, None, :])[:, -1, :]
        J = self._reach_score(z1, w_rep).reshape(B, N)
        best = jnp.argmax(J, axis=1)
        a_mpc = chunks.reshape(B, N, cd)[jnp.arange(B), best]
        return a_mpc, J.max(axis=1).mean()

    @jax.jit
    def update_low_distill(self, z, w_latent, a_mpc):
        """Actor-only plain BC-flow of the LL toward a_mpc, conditioned on the
        subgoal derived from w. V/HL get zero gradient (frozen)."""
        new_rng, rng = jax.random.split(self.rng)
        sg = self._subgoal_from_latent(z, w_latent)

        def loss_fn(grad_params):
            obs_cond = jnp.concatenate([z, sg], axis=-1)
            loss, bc = self._flow_bc('low_actor_bc_flow', obs_cond, a_mpc, rng,
                                     jnp.ones((z.shape[0],)), grad_params)
            return loss, {'distill_loss': loss, 'distill_bc': bc}

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update_low_anchored(self, z, w_latent, real_chunk, a_mpc):
        """ANCHORED LL distillation -- the fix for the Phase-2 collapse. Mostly
        plain flow-BC toward the REAL data chunk (the Phase-1 signal that gives the
        ~88% baseline -- the data manifold anchor), plus ll_mpc_coef * flow-BC
        toward the MPC chunk a_mpc (the grounding nudge). This mirrors the working
        flat method's '~99% real BC + minority MPC' recipe; ll_mpc_coef=0 reproduces
        a pure Phase-1 continuation. V/HL get zero gradient (frozen)."""
        new_rng, rng = jax.random.split(self.rng)
        sg = self._subgoal_from_latent(z, w_latent)
        ones = jnp.ones((z.shape[0],))

        def loss_fn(grad_params):
            obs_cond = jnp.concatenate([z, sg], axis=-1)
            rng_a, rng_m = jax.random.split(rng)
            anchor, abc = self._flow_bc('low_actor_bc_flow', obs_cond, real_chunk, rng_a, ones, grad_params)
            nudge, mbc = self._flow_bc('low_actor_bc_flow', obs_cond, a_mpc, rng_m, ones, grad_params)
            loss = anchor + self.config['ll_mpc_coef'] * nudge
            return loss, {'distill_loss': loss, 'anchor_bc': abc, 'mpc_bc': mbc, 'distill_bc': mbc}

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(network=new_network, rng=new_rng), info

    def roll_low_in_wm(self, z, subgoal, seed):
        """Closed-loop LL rollout toward a subgoal (already in subgoal space) for
        subgoal_steps env steps; returns z_achieved."""
        n_chunks = self.config['subgoal_steps'] // self.config['action_chunk_len']
        zc = z
        for _ in range(n_chunks):
            seed, k = jax.random.split(seed)
            chunk = self.sample_low_chunk(zc, subgoal, seed=k)
            zc = self.wm_model.apply(self.wm_params, zc[:, None, :], chunk[:, None, :])[:, -1, :]
        return zc

    @jax.jit
    def update_high_ground(self, z, g, w_data_latent, seed):
        """Actor-only AWR grounding of the HL by the value the LL ACHIEVES when
        aiming for each candidate subgoal. Anchor toward the data subgoal keeps
        proposals on-manifold."""
        N = self.config['hl_num_samples']
        new_rng, rng = jax.random.split(self.rng)

        def _cand(key):
            sk = self.sample_high_subgoal(z, g, seed=key)        # subgoal-space candidate
            z_ach = self.roll_low_in_wm(z, sk, jax.random.split(key)[0])
            v1, v2 = self._value(z_ach, g)
            return sk, (v1 + v2) / 2.0

        ws, Js = jax.vmap(_cand)(jax.random.split(seed, N))      # (N,B,sg), (N,B)
        ws = jax.lax.stop_gradient(ws)
        Js = jax.lax.stop_gradient(Js)
        adv = Js - Js.mean(axis=0, keepdims=True)
        u = jax.nn.softmax(self.config['high_alpha'] * adv, axis=0)
        anchor_target = self._subgoal_from_latent(z, w_data_latent)

        def loss_fn(grad_params):
            obs_cond = jnp.concatenate([z, g], axis=-1)
            bc_keys = jax.random.split(rng, N + 1)
            anchor, _ = self._flow_bc('high_actor_flow', obs_cond, anchor_target, bc_keys[0],
                                      jnp.ones((z.shape[0],)), grad_params)
            rwr = 0.0
            for i in range(N):
                li, _ = self._flow_bc('high_actor_flow', obs_cond, ws[i], bc_keys[i + 1],
                                      u[i], grad_params)
                rwr = rwr + li
            loss = self.config['hl_anchor_coef'] * anchor + rwr
            return loss, {'hl_ground_loss': loss, 'hl_anchor': anchor,
                          'hl_J_mean': Js.mean(), 'hl_adv_std': adv.std(), 'hl_w_max': u.max()}

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update_high_ground_real(self, z, g, cand_latents, w_data_latent, seed):
        """HL grounding with REAL candidate subgoals.

        Scores K real future latents from the trajectory by rolling the LL toward
        each in the WM and taking V(z_ach, g). Each update step, stochastically
        picks ONE BC target: with probability hl_ground_coef use the argmax winner,
        otherwise use the data anchor (cand_latents[0] = 1×subgoal_steps).

        One target per step is critical: training a flow model toward TWO targets
        simultaneously from the same conditioning [z,g] forces it to predict a
        weighted average trajectory, which lands between the targets in the 192-D
        space. That interpolated latent is almost certainly off the data manifold,
        causing LL failure. Stochastic selection keeps every BC target on-manifold."""
        K = cand_latents.shape[0]
        new_rng, rng = jax.random.split(self.rng)

        n_avg = self.config.get('hl_score_avg', 3)

        score_mode = self.config.get('hl_score_mode', 'reachability')

        def _score(w_lat, keys):
            sg = self._subgoal_from_latent(z, w_lat)
            def _one(key):
                z_ach = self.roll_low_in_wm(z, sg, key)
                v1, v2 = self._value(z_ach, g)
                return (v1 + v2) / 2.0
            J_rollout = jax.vmap(_one)(keys).mean(axis=0)   # (B,)
            if score_mode == 'reachability':
                # Subtract candidate's own V to correct for value inversion:
                # far candidates have trivially higher V(cand,g) despite being
                # unreachable; the gap J_rollout - V_cand penalises them.
                v1c, v2c = self._value(w_lat, g)
                V_cand = (v1c + v2c) / 2.0
                return J_rollout - V_cand
            else:  # 'rollout' — original: just V(z_ach, g) from WM rollout
                return J_rollout

        score_keys = jax.random.split(seed, K * n_avg).reshape(K, n_avg, 2)
        Js = jax.vmap(_score)(cand_latents, score_keys)      # (K,B)
        Js = jax.lax.stop_gradient(Js)
        best = jnp.argmax(Js, axis=0)                                    # (B,)
        win_lat = jnp.take_along_axis(cand_latents, best[None, :, None], axis=0)[0]
        win_target = jax.lax.stop_gradient(self._subgoal_from_latent(z, win_lat))
        anchor_target = jax.lax.stop_gradient(self._subgoal_from_latent(z, w_data_latent))

        # Stochastically pick ONE target per batch element: winner with probability
        # hl_ground_coef, anchor otherwise. Avoids conflicting gradient signals.
        rng, coin = jax.random.split(rng)
        use_winner = jax.random.uniform(coin, (z.shape[0],)) < self.config['hl_ground_coef']
        target = jnp.where(use_winner[:, None], win_target, anchor_target)
        target = jax.lax.stop_gradient(target)
        ones = jnp.ones((z.shape[0],))

        def loss_fn(grad_params):
            obs_cond = jnp.concatenate([z, g], axis=-1)
            k1, _ = jax.random.split(rng)
            loss, bc = self._flow_bc('high_actor_flow', obs_cond, target, k1, ones, grad_params)
            J_data = Js[0]
            return loss, {'hl_ground_loss': loss, 'hl_bc': bc,
                          'hl_J_mean': Js.mean(), 'hl_J_best': Js.max(axis=0).mean(),
                          'hl_J_data': J_data.mean(),
                          'hl_gain': (Js.max(axis=0) - J_data).mean(),
                          'hl_pick_data_frac': (best == 0).mean(),
                          'hl_use_winner_frac': use_winner.mean(),
                          'win_lat': win_lat}

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update_high_ground_awr(self, z, g, w_data_latent, seed):
        """HL grounding via AWR on REAL data subgoals weighted by WM-ROLLOUT value.

        The ONLY BC target is the real data subgoal w_data (always on-manifold).
        Each (z, g, w_data) tuple is reweighted by the advantage of the state the LL
        actually REACHES when chasing w_data in the WM:

            adv = mean_k V(roll_low_in_wm(z, sg_k), g) - V(z, g)
            w   = min(exp(hl_awr_alpha * adv), 100)

        This is Phase-1's AWR high-actor loss with the 1-step value advantage replaced
        by the LL-reachability (WM-rollout) advantage. NO self-samples are ever fed
        back into the flow -> no self-consuming drift; weighting (not unweighted BC)
        preserves the AWR sharpening -> no de-sharpening. Both are the failure modes
        that made update_high_ground_real degrade V_roll on CPU."""
        new_rng, rng = jax.random.split(self.rng)
        n_avg = self.config.get('hl_score_avg', 3)
        sg = self._subgoal_from_latent(z, w_data_latent)        # LL conditioning subgoal

        score_keys = jax.random.split(seed, n_avg)

        def _one(key):
            z_ach = self.roll_low_in_wm(z, sg, key)
            v1, v2 = self._value(z_ach, g)
            return (v1 + v2) / 2.0

        J_roll = jax.vmap(_one)(score_keys).mean(axis=0)        # (B,)
        v1z, v2z = self._value(z, g)
        Vz = (v1z + v2z) / 2.0
        adv = J_roll - Vz                                       # (B,)
        # Standardise the advantage within the batch so hl_awr_alpha is meaningful
        # regardless of the (large, negative) value scale and the weights don't all
        # saturate at the cap. Batch-mean baseline = clean within-batch selection.
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        adv = jax.lax.stop_gradient(adv)
        alpha = self.config.get('hl_awr_alpha', self.config['high_alpha'])
        weight = jnp.minimum(jnp.exp(alpha * adv), 100.0)
        target = jax.lax.stop_gradient(sg)

        def loss_fn(grad_params):
            obs_cond = jnp.concatenate([z, g], axis=-1)
            k1, _ = jax.random.split(rng)
            loss, bc = self._flow_bc('high_actor_flow', obs_cond, target, k1, weight, grad_params)
            return loss, {'hl_ground_loss': loss, 'hl_bc': bc,
                          'hl_J_mean': J_roll.mean(), 'hl_J_best': J_roll.mean(),
                          'hl_adv_mean': adv.mean(), 'hl_adv_std': adv.std(),
                          'hl_w_max': weight.max(), 'hl_w_mean': weight.mean(),
                          'hl_gain': adv.mean(), 'hl_pick_data_frac': 1.0,
                          'hl_use_winner_frac': 0.0}

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update_high_ground_phase1awr(self, z, g, w_data_latent, seed):
        """HL AWR using Phase-1's own advantage signal V(w_data,g)−V(z,g) instead of
        the WM-rollout advantage. Ablation to isolate whether degradation comes from
        (a) the noisy WM rollout signal, or (b) the BC target distribution itself.
        If this degrades at the same rate as update_high_ground_awr, the BC target
        (data subgoals being broader than the Phase-1 HL distribution) is the root
        cause and no signal improvement can fix it."""
        new_rng, rng = jax.random.split(self.rng)
        sg = self._subgoal_from_latent(z, w_data_latent)

        # Phase-1-style advantage: V(w_data, g) − V(z, g). No WM rollout.
        v1w, v2w = self._value(sg, g)
        J_phase1 = (v1w + v2w) / 2.0
        v1z, v2z = self._value(z, g)
        Vz = (v1z + v2z) / 2.0
        adv = J_phase1 - Vz                                       # (B,)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        adv = jax.lax.stop_gradient(adv)
        alpha = self.config.get('hl_awr_alpha', self.config['high_alpha'])
        weight = jnp.minimum(jnp.exp(alpha * adv), 100.0)
        target = jax.lax.stop_gradient(sg)

        def loss_fn(grad_params):
            obs_cond = jnp.concatenate([z, g], axis=-1)
            k1, _ = jax.random.split(rng)
            loss, bc = self._flow_bc('high_actor_flow', obs_cond, target, k1, weight, grad_params)
            return loss, {'hl_ground_loss': loss, 'hl_bc': bc,
                          'hl_J_mean': J_phase1.mean(), 'hl_J_best': J_phase1.max(),
                          'hl_adv_mean': adv.mean(), 'hl_adv_std': adv.std(),
                          'hl_w_max': weight.max(), 'hl_w_mean': weight.mean(),
                          'hl_gain': adv.mean(), 'hl_pick_data_frac': 1.0,
                          'hl_use_winner_frac': 0.0}

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update_high_ground_hrf(self, z, g, seed):
        """Hindsight Reality Forcing (HRF): BC toward ACHIEVED latent states, not proposals.

        This breaks both failure modes:
          - Value gaming: HL cannot propose a near-g hallucination and get positive credit,
            because the BC target is z_ach (where the LL physically ends up in the WM), not
            w_cand (what the HL proposed). Even if HL proposes w_cand near g, the frozen LL
            cannot traverse there, so z_ach has low V and gets zero/negative AWR weight.
          - De-sharpening: z_ach values are constrained to the achievable manifold
            (states reachable by the frozen LL from z). This manifold is narrower than the
            data-subgoal distribution, so BC toward z_ach cannot broaden the HL.

        Algorithm:
          1. Sample K subgoals from the current HL: sg_k = HL(z, g, noise_k)
          2. Roll frozen LL toward each in WM: z_ach_k = LL_WM(z, sg_k)
          3. Advantage vs external baseline: adv_k = V(z_ach_k, g) - V(z, g)
          4. Standardize across all K*B elements; AWR weight: w_k = exp(alpha * adv_norm)
          5. Flow BC loss: sum_k [flow_bc(HL; obs_cond, z_ach_k, weight=w_k)] / K

        The BC target is z_ach (not sg). Using z_ach as the HL output target means
        the HL learns to directly propose the achievable states that have high V, rather
        than proposals that may lead the LL there indirectly. This is stable because
        z_ach is already reachable by definition."""
        new_rng, rng = jax.random.split(self.rng)
        K     = self.config.get('hl_num_samples', 8)
        alpha = self.config.get('hl_awr_alpha', self.config['high_alpha'])
        B     = z.shape[0]

        # Sample K subgoals from the current HL; roll LL toward each in WM.
        def _cand(key):
            sg    = self.sample_high_subgoal(z, g, seed=key)
            rkey, _ = jax.random.split(key)
            z_ach = self.roll_low_in_wm(z, sg, rkey)            # (B, obs_dim)
            v1, v2 = self._value(z_ach, g)
            return z_ach, (v1 + v2) / 2.0                        # (B, obs_dim), (B,)

        cand_keys     = jax.random.split(seed, K)
        z_achs, Js    = jax.vmap(_cand)(cand_keys)               # (K,B,obs), (K,B)
        z_achs = jax.lax.stop_gradient(z_achs)
        Js     = jax.lax.stop_gradient(Js)

        # External baseline V(z, g): advantage > 0 only if LL improved over doing nothing.
        v1z, v2z = self._value(z, g)
        Vz = jax.lax.stop_gradient((v1z + v2z) / 2.0)           # (B,)

        adv = Js - Vz[None, :]                                   # (K, B), external baseline
        # Global standardise across all K*B elements so alpha is scale-independent.
        adv_flat = adv.reshape(-1)
        adv_norm = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
        adv_norm = adv_norm.reshape(K, B)
        weight_k = jnp.minimum(jnp.exp(alpha * adv_norm), 100.0) # (K, B)

        def loss_fn(grad_params):
            obs_cond  = jax.lax.stop_gradient(jnp.concatenate([z, g], axis=-1))
            bc_keys   = jax.random.split(rng, K)
            total = 0.0
            for k in range(K):
                lk, _ = self._flow_bc('high_actor_flow', obs_cond,
                                      z_achs[k], bc_keys[k], weight_k[k], grad_params)
                total = total + lk
            loss = total / K
            return loss, {
                'hl_ground_loss': loss,
                'hl_J_mean':      Js.mean(),
                'hl_J_best':      Js.max(axis=0).mean(),
                'hl_adv_mean':    adv.mean(),
                'hl_adv_std':     adv.std(),
                'hl_w_max':       weight_k.max(),
                'hl_w_mean':      weight_k.mean(),
                'hl_gain':        adv.mean(),
                'hl_pick_data_frac': 0.0,
                'hl_use_winner_frac': 0.0,
            }

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update_high_ground_dpo(self, z, g, ref_params, seed):
        """Flow-DPO HL update with Phase-1 reference policy + feasibility penalty.

        Samples K subgoals from the current HL, scores each by
            J_safe = V(z_ach, g) - lambda * ||z_ach - sg||
        where the penalty catches value gaming (a subgoal V rates highly but the LL
        cannot reach will have large ||z_ach - sg||). Picks winner at the 75th
        percentile and loser at the 25th (not argmax/argmin — avoids OOD extremes).

        DPO loss with reference policy:
            logits = beta * [(L_ref(win) - L_curr(win)) - (L_ref(lose) - L_curr(lose))]
            loss   = -log_sigmoid(logits).mean()
        This is a RELATIVE update: it does not push the HL toward an absolute target
        (the de-sharpening failure mode of AWR BC). The reference policy (Phase-1 HL
        frozen at startup) bounds KL drift per step."""
        new_rng, rng = jax.random.split(self.rng)
        K       = self.config.get('hl_num_samples', 8)
        lambda_ = self.config.get('hl_dpo_feasibility', 0.1)
        beta    = self.config.get('hl_dpo_beta', 0.5)
        B       = z.shape[0]

        # Sample K subgoals from the current HL; score each with WM rollout + feasibility.
        def _cand(key):
            sg = self.sample_high_subgoal(z, g, seed=key)
            roll_key, _ = jax.random.split(key)
            z_ach = self.roll_low_in_wm(z, sg, roll_key)
            v1, v2 = self._value(z_ach, g)
            J      = (v1 + v2) / 2.0                            # (B,)
            dist   = jnp.linalg.norm(z_ach - sg, axis=-1)       # (B,)
            return sg, J - lambda_ * dist                        # J_safe: (B,)

        cand_keys = jax.random.split(seed, K)
        ws, Js = jax.vmap(_cand)(cand_keys)                      # (K,B,sg), (K,B)
        ws = jax.lax.stop_gradient(ws)
        Js = jax.lax.stop_gradient(Js)

        # Percentile winner/loser: 75th and 25th over the K candidates.
        # For K=8: win_k=5, lose_k=1 (0-indexed, ascending sort).
        sorted_idx = jnp.argsort(Js, axis=0)                     # (K,B) ascending
        win_k  = int(0.75 * (K - 1))
        lose_k = int(0.25 * (K - 1))
        sg_dim = ws.shape[-1]

        def _gather(k):
            idx = jnp.broadcast_to(sorted_idx[k][None, :, None], (1, B, sg_dim))
            return jnp.take_along_axis(ws, idx, axis=0).squeeze(0)  # (B, sg_dim)

        win_lat  = jax.lax.stop_gradient(_gather(win_k))         # (B, sg_dim)
        lose_lat = jax.lax.stop_gradient(_gather(lose_k))        # (B, sg_dim)

        # Same noise seed for winner and loser: fair comparison (same x_0 and t draw).
        k_bc, _ = jax.random.split(rng)
        obs_cond = jax.lax.stop_gradient(jnp.concatenate([z, g], axis=-1))

        def loss_fn(grad_params):
            # Current policy per-sample losses
            per_win_curr  = self._flow_bc_per_sample(
                'high_actor_flow', obs_cond, win_lat,  k_bc, grad_params)  # (B,)
            per_lose_curr = self._flow_bc_per_sample(
                'high_actor_flow', obs_cond, lose_lat, k_bc, grad_params)  # (B,)
            # Reference (Phase-1) per-sample losses — stop_gradient: no backprop through ref.
            per_win_ref  = jax.lax.stop_gradient(self._flow_bc_per_sample(
                'high_actor_flow', obs_cond, win_lat,  k_bc, ref_params))  # (B,)
            per_lose_ref = jax.lax.stop_gradient(self._flow_bc_per_sample(
                'high_actor_flow', obs_cond, lose_lat, k_bc, ref_params))  # (B,)
            # DPO logits: how much more the current policy improved on the winner vs loser.
            # Higher logit = current is proportionally better at winner than reference, and
            # worse at loser than reference — this is what the DPO gradient drives toward.
            improve_win  = per_win_ref  - per_win_curr    # (B,): positive = curr better at win
            improve_lose = per_lose_ref - per_lose_curr   # (B,)
            logits = beta * (improve_win - improve_lose)  # (B,)
            loss   = -jax.nn.log_sigmoid(logits).mean()
            J_win  = jnp.take_along_axis(
                Js, sorted_idx[win_k:win_k+1], axis=0).squeeze(0).mean()
            J_lose = jnp.take_along_axis(
                Js, sorted_idx[lose_k:lose_k+1], axis=0).squeeze(0).mean()
            return loss, {
                'hl_ground_loss':   loss,
                'hl_dpo_logits':    logits.mean(),
                'hl_improve_win':   improve_win.mean(),
                'hl_improve_lose':  improve_lose.mean(),
                'hl_J_mean':        Js.mean(),
                'hl_J_win':         J_win,
                'hl_J_lose':        J_lose,
            }

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(network=new_network, rng=new_rng), info

    # ------------------------------------------------------------------ #
    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config, wm_model=None, wm_params=None):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        obs_dim = ex_observations.shape[-1]
        action_dim = ex_actions.shape[-1]
        chunk_dim = action_dim * config['action_chunk_len']
        ss = config['subgoal_space']
        subgoal_dim = config['rep_dim'] if ss == 'rep' else obs_dim

        ex_goals = ex_observations
        ex_subgoal = jnp.zeros((*ex_observations.shape[:-1], subgoal_dim), dtype=jnp.float32)
        ex_chunk = jnp.zeros((*ex_observations.shape[:-1], chunk_dim), dtype=jnp.float32)
        ex_times = jnp.zeros((*ex_observations.shape[:-1], 1), dtype=jnp.float32)
        ex_low_cond = jnp.concatenate([ex_observations, ex_subgoal], axis=-1)
        ex_high_cond = jnp.concatenate([ex_observations, ex_goals], axis=-1)

        network_info = {}
        if ss == 'rep':
            # goal_rep phi([z;g]) -> length-normalized rep; value bottlenecks through it.
            def _rep():
                return nn.Sequential([MLP((*config['value_hidden_dims'], config['rep_dim']),
                                          activate_final=False, layer_norm=config['layer_norm']),
                                      LengthNormalize()])
            ex_rep = jnp.zeros((*ex_observations.shape[:-1], config['rep_dim']), dtype=jnp.float32)
            value_def = GCValue(config['value_hidden_dims'], config['layer_norm'], True, gc_encoder=None)
            target_value_def = GCValue(config['value_hidden_dims'], config['layer_norm'], True, gc_encoder=None)
            network_info['goal_rep'] = (_rep(), (ex_high_cond,))
            network_info['target_goal_rep'] = (_rep(), (ex_high_cond,))
            network_info['value'] = (value_def, (ex_observations, ex_rep))
            network_info['target_value'] = (target_value_def, (ex_observations, ex_rep))
        else:
            value_def = GCValue(config['value_hidden_dims'], config['layer_norm'], True, gc_encoder=None)
            target_value_def = GCValue(config['value_hidden_dims'], config['layer_norm'], True, gc_encoder=None)
            network_info['value'] = (value_def, (ex_observations, ex_goals))
            network_info['target_value'] = (target_value_def, (ex_observations, ex_goals))

        low_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'], action_dim=chunk_dim,
            layer_norm=config['actor_layer_norm'], encoder=None,
            use_fourier_features=config['use_fourier_features'],
            fourier_feature_dim=config['fourier_feature_dim'])
        high_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'], action_dim=subgoal_dim,
            layer_norm=config['actor_layer_norm'], encoder=None,
            use_fourier_features=config['use_fourier_features'],
            fourier_feature_dim=config['fourier_feature_dim'])
        network_info['low_actor_bc_flow'] = (low_flow_def, (ex_low_cond, ex_chunk, ex_times))
        network_info['high_actor_flow'] = (high_flow_def, (ex_high_cond, ex_subgoal, ex_times))

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)
        params = network.params
        params['modules_target_value'] = params['modules_value']
        if ss == 'rep':
            params['modules_target_goal_rep'] = params['modules_goal_rep']

        config['action_dim'] = action_dim
        config['subgoal_dim'] = subgoal_dim
        if config['wgsp_coef'] > 0.0:
            assert wm_model is not None and wm_params is not None, 'wgsp_coef>0 needs a frozen WM'
        return cls(rng, network=network, config=flax.core.FrozenDict(**config),
                   wm_model=wm_model, wm_params=wm_params)


def get_config():
    return ml_collections.ConfigDict(dict(
        agent_name='wgsp_lsg',
        lr=3e-4,
        batch_size=1024,
        actor_hidden_dims=(512, 512, 512),
        value_hidden_dims=(512, 512, 512),
        layer_norm=True,
        actor_layer_norm=False,
        discount=0.99,
        tau=0.005,
        expectile=0.7,
        low_alpha=3.0,
        high_alpha=3.0,
        ll_awr=False,
        subgoal_space='latent',  # ABLATION: 'latent' (192-D) or 'rep' (10-D HIQL-style).
        rep_dim=10,
        subgoal_steps=10,
        action_chunk_len=5,
        flow_steps=10,
        use_fourier_features=False,
        fourier_feature_dim=64,
        # Phase-2 (WGSP-MPC) knobs; 0 -> Phase-1 only (no WM traced).
        wgsp_coef=0.0,
        wgsp_num_samples=32,
        wgsp_fmq_eta=0.0,
        ll_reach_metric='distance',  # 'distance' (latent ||z'-w||) or 'value' (V(z',w)).
        hl_num_samples=8,
        hl_anchor_coef=1.0,
        hl_ground_coef=0.3,  # weight of the real-candidate grounding nudge vs the data anchor.
        hl_score_avg=3,      # rollout samples averaged per candidate to reduce scoring noise.
        hl_score_mode='reachability',  # 'reachability': J_rollout - V_cand; 'rollout': J_rollout only.
        hl_awr_alpha=1.0,    # [hl_mode=awr] AWR temperature on the WM-rollout advantage.
        ll_mpc_coef=0.3,     # weight of the MPC nudge vs the real-data BC anchor in the LL.
        action_dim=ml_collections.config_dict.placeholder(int),
        subgoal_dim=ml_collections.config_dict.placeholder(int),
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
        encoder=ml_collections.config_dict.placeholder(str),
    ))
