import copy
import functools
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value

class ACFQLAgent(flax.struct.PyTreeNode):
    """Flow Q-learning (FQL) agent with action chunking.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()
    # Optional differentiable-rollout support. When ``wm_params`` is not None,
    # the actor loss adds an analytic-policy-gradient term computed by rolling
    # out the BC flow's actions through a frozen JAX WM. These are
    # ``nonpytree_field`` so they ride along on the instance but are NOT
    # serialised in the pytree state_dict -- otherwise old offline checkpoints
    # would fail to restore due to missing fields. JIT recompiles when these
    # change shape/identity, which is fine because we set them once at boot.
    wm_params: Any = nonpytree_field(default=None)
    z_goal: Any = nonpytree_field(default=None)
    wm_model: Any = nonpytree_field(default=None)
    # Optional joint-training WM TrainState. When set (and config["use_joint_wm"]
    # is True), the WM's parameters update alongside the actor/critic during the
    # offline phase, using a two-term loss: L_pred (multi-step prediction MSE,
    # scale-invariant) + beta * L_value (Bellman residual via WM-predicted z').
    # Distinct from ``wm_params`` above (which is frozen, for diff-rollouts).
    wm_state: Any = None

    def critic_loss(self, batch, grad_params, rng):
        """Compute the FQL critic loss."""

        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :] # take the first action
        
        # TD loss
        rng, sample_rng = jax.random.split(rng)
        next_actions = self.sample_actions(batch['next_observations'][..., -1, :], rng=sample_rng)

        next_qs = self.network.select(f'target_critic')(batch['next_observations'][..., -1, :], actions=next_actions)
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)
        
        target_q = batch['rewards'][..., -1] + \
            (self.config['discount'] ** self.config["horizon_length"]) * batch['masks'][..., -1] * next_q

        q = self.network.select('critic')(batch['observations'], actions=batch_actions, params=grad_params)
        
        critic_loss = (jnp.square(q - target_q) * batch['valid'][..., -1]).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def differentiable_rollout_loss(self, batch, grad_params, rng):
        """Analytic-policy-gradient term: roll out BC-flow actions through a
        frozen JAX WM, score with dense reward, return negative mean return.

        Single Python for-loop over H steps. At each step we sample a noise
        vector, integrate the BC flow (uses grad_params -> gradient flows back
        into actor params), then advance the state via the frozen WM. The
        per-step latent is scored against z_goal with the same dense reward
        shape used by WMEnv.
        """
        H = int(self.config.get("rollout_horizon", 3))
        scale = float(self.config.get("rollout_dense_scale", 10.0))
        gamma = float(self.config["discount"])
        action_dim = int(self.config["action_dim"]) * (
            int(self.config["horizon_length"]) if self.config["action_chunking"] else 1
        )

        z = batch["observations"]
        B = z.shape[0]
        rng, *step_rngs = jax.random.split(rng, H + 1)

        total_return = jnp.zeros(B)
        discount = 1.0
        z_t = z
        for k in range(H):
            n = jax.random.normal(step_rngs[k], (B, action_dim))
            a = self._compute_flow_actions_diff(z_t, n, grad_params)
            a = jnp.clip(a, -1.0, 1.0)
            z_next = self.wm_model.apply(
                self.wm_params, z_t[:, None, :], a[:, None, :]
            )[:, -1, :]
            d = jnp.linalg.norm(z_next - self.z_goal[None, :], axis=-1)
            r = -d / scale
            total_return = total_return + discount * r
            discount = discount * gamma
            z_t = z_next

        # Negative for gradient ASCENT in the actor optimizer
        return -total_return.mean()

    def _compute_flow_actions_diff(self, observations, noises, grad_params):
        """Like compute_flow_actions but routes through grad_params so the
        gradient flows back into actor params. We let actor_bc_flow run its
        own internal encoder (is_encoded=False) since that's the same path
        actor_loss uses with params=grad_params and is the verified pattern."""
        actions = noises
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_bc_flow')(
                observations, actions, t, params=grad_params)
            actions = actions + vels / self.config['flow_steps']
        return actions

    def actor_loss(self, batch, grad_params, rng):
        """Compute the FQL actor loss."""
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))  # fold in horizon_length together with action_dim
        else:
            batch_actions = batch["actions"][..., 0, :] # take the first one
        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # BC flow loss.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch_actions
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_bc_flow')(batch['observations'], x_t, t, params=grad_params)

        # only bc on the valid chunk indices
        if self.config["action_chunking"]:
            bc_flow_loss = jnp.mean(
                jnp.reshape(
                    (pred - vel) ** 2, 
                    (batch_size, self.config["horizon_length"], self.config["action_dim"]) 
                ) * batch["valid"][..., None]
            )
        else:
            bc_flow_loss = jnp.mean(jnp.square(pred - vel))

        if self.config["actor_type"] == "distill-ddpg":
            # Distillation loss.
            rng, noise_rng = jax.random.split(rng)
            noises = jax.random.normal(noise_rng, (batch_size, action_dim))
            target_flow_actions = self.compute_flow_actions(batch['observations'], noises=noises)
            actor_actions = self.network.select('actor_onestep_flow')(batch['observations'], noises, params=grad_params)
            distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)
            
            # Q loss.
            actor_actions = jnp.clip(actor_actions, -1, 1)

            qs = self.network.select(f'critic')(batch['observations'], actions=actor_actions)
            q = jnp.mean(qs, axis=0)
            q_loss = -q.mean()
        else:
            distill_loss = jnp.zeros(())
            q_loss = jnp.zeros(())

        # Optional differentiable-rollout term (analytic policy gradient
        # through a frozen JAX WM). Only active when wm_params is set.
        rollout_loss = jnp.zeros(())
        if self.wm_params is not None and self.config.get("rollout_loss_weight", 0.0) > 0.0:
            rng, rollout_rng = jax.random.split(rng)
            rollout_loss = self.differentiable_rollout_loss(batch, grad_params, rollout_rng)

        # Total loss.
        actor_loss = (
            bc_flow_loss
            + self.config['alpha'] * distill_loss
            + q_loss
            + self.config.get("rollout_loss_weight", 0.0) * rollout_loss
        )

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'distill_loss': distill_loss,
            'rollout_loss': rollout_loss,
        }

    def wm_loss(self, batch, grad_params, wm_grad_params, rng):
        """Joint-training WM loss = alpha * L_pred + beta * L_value.

        L_pred:  one-step prediction MSE, scale-invariant -- the WM has to be
                 a good physical predictor.
        L_value: Bellman residual using the TARGET critic at both real (z, a)
                 and WM-predicted z'. Target params break the WM-critic
                 circular dependency.

        Only the WM params receive gradient from this term. The actor/critic
        live params are constants here (we use target_critic which lives at
        ``self.network.params['modules_target_critic']``).
        """
        # Batches have shape (B, T, ...) with T == sample_seq_len. In the
        # E (WM) pipeline T==1.
        z = batch['observations']                    # (B, D)
        a = batch['actions'][..., 0, :]              # (B, A)
        z_next_real = batch['next_observations'][..., -1, :]   # (B, D)
        r = batch['rewards'][..., -1]                # (B,)
        masks = batch['masks'][..., -1]              # (B,)

        # === L_pred: one-step, scale-invariant ===
        # wm_state.apply_fn is bound LeJEPAJaxForward.apply; takes
        # (variables_dict, emb, action_raw) and expects (B, T, D)/(B, T, A) inputs.
        z_pred = self.wm_state.apply_fn(
            {'params': wm_grad_params}, z[:, None, :], a[:, None, :]
        )[:, -1, :]
        sq_err = jnp.sum((z_pred - z_next_real) ** 2, axis=-1)
        norm_sq = jnp.sum(z_next_real ** 2, axis=-1) + 1e-6
        pred_loss = jnp.mean(sq_err / norm_sq)

        # === L_value: Bellman residual ===
        # Target Q at REAL (z, a) -- this is a frozen target (no grad).
        target_q_real = self.network.select('target_critic')(z, actions=a).mean(axis=0)

        # Bootstrap action at the WM-predicted next state. We sample using the
        # live actor for relevance, but stop_gradient on z_pred for the action
        # sampling step so the action does not carry gradient (action sampling
        # is not differentiable through best-of-n argmax anyway).
        rng, sample_rng = jax.random.split(rng)
        next_a_pred = self.sample_actions(jax.lax.stop_gradient(z_pred), rng=sample_rng)

        # V(z_pred) via target critic. Gradient flows back through z_pred into
        # WM params (the key step that makes the WM "value-aware").
        next_v_pred = self.network.select('target_critic')(z_pred, actions=next_a_pred).mean(axis=0)

        bellman_target = r + self.config['discount'] * masks * next_v_pred
        value_loss = jnp.mean(jnp.square(target_q_real - bellman_target))

        return pred_loss, value_loss

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss (no joint WM training)."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def total_loss_joint(self, batch, grad_params, wm_grad_params, rng):
        """Total loss WITH joint WM training term.

        Actor and critic losses are unchanged (use the real next_observations
        from the offline buffer). The WM loss adds:
            alpha * L_pred + beta(step) * L_value
        where beta linearly ramps from 0 to ``joint_beta`` over
        ``joint_beta_ramp_steps`` steps -- gives the critic time to settle
        before the WM starts listening to it.
        """
        info = {}
        rng, actor_rng, critic_rng, wm_rng = jax.random.split(rng, 4)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        pred_loss, value_loss = self.wm_loss(batch, grad_params, wm_grad_params, wm_rng)

        alpha = self.config.get("joint_alpha", 1.0)
        beta = self.config.get("joint_beta", 0.1)
        ramp_steps = self.config.get("joint_beta_ramp_steps", 50000)
        # network.step is a plain int when called outside JIT, a JAX tracer
        # when called inside it. jnp.asarray works for both.
        step = jnp.asarray(self.network.step, dtype=jnp.float32)
        beta_ramped = beta * jnp.minimum(1.0, step / float(ramp_steps))

        wm_total = alpha * pred_loss + beta_ramped * value_loss
        loss = critic_loss + actor_loss + wm_total

        info['wm/pred_loss'] = pred_loss
        info['wm/value_loss'] = value_loss
        info['wm/beta_ramped'] = beta_ramped
        info['wm/total'] = wm_total
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @staticmethod
    def _update(agent, batch):
        """Update the agent and return a new agent with information dictionary.

        Branches on whether joint WM training is enabled. When ``agent.wm_state``
        is ``None`` we run the original single-optimiser path. When it's a
        TrainState we additionally compute gradients for the WM and step its
        optimiser. The branch is decided at trace time (structural property
        of the agent pytree), so each path compiles independently.
        """
        new_rng, rng = jax.random.split(agent.rng)

        if agent.wm_state is None:
            def loss_fn(grad_params):
                return agent.total_loss(batch, grad_params, rng=rng)
            new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
            agent.target_update(new_network, 'critic')
            return agent.replace(network=new_network, rng=new_rng), info

        # Joint WM + policy update
        def joint_loss_fn(grad_params, wm_grad_params):
            return agent.total_loss_joint(batch, grad_params, wm_grad_params, rng=rng)
        grad_fn = jax.grad(joint_loss_fn, has_aux=True, argnums=(0, 1))
        (net_grads, wm_grads), info = grad_fn(agent.network.params, agent.wm_state.params)
        new_network = agent.network.apply_gradients(grads=net_grads)
        new_wm_state = agent.wm_state.apply_gradients(grads=wm_grads)
        agent.target_update(new_network, 'critic')
        return agent.replace(network=new_network, wm_state=new_wm_state, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)
    
    @jax.jit
    def batch_update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        # update_size = batch["observations"].shape[0]
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)
    
    @jax.jit
    def sample_actions(
        self,
        observations,
        rng=None,
    ):
        
        if self.config["actor_type"] == "distill-ddpg":
            noises = jax.random.normal(
                rng,
                (
                    *observations.shape[: -len(self.config['ob_dims'])],  # batch_size
                    self.config['action_dim'] * \
                        (self.config['horizon_length'] if self.config["action_chunking"] else 1),
                ),
            )
            actions = self.network.select(f'actor_onestep_flow')(observations, noises)
            actions = jnp.clip(actions, -1, 1)

        elif self.config["actor_type"] == "best-of-n":
            action_dim = self.config['action_dim'] * \
                        (self.config['horizon_length'] if self.config["action_chunking"] else 1)
            N = self.config["actor_num_samples"]

            # Image encoder: ob_dims has more than 2 elements (batch + H + W + C).
            # For 1-D latent obs ob_dims is (batch, latent_dim), len == 2.
            image_encoder = (self.config['encoder'] is not None and
                             len(self.config['ob_dims']) > 2)

            if image_encoder:
                # Pre-encode images → 1-D latents for the ACTOR only (avoids N CNN passes).
                # The CRITIC must use its own encoder; we call it via a reshape trick below
                # so no parameter-sharing conflicts arise from registering the encoder twice.
                actor_obs = self.network.select('actor_bc_flow_encoder')(observations)
                # actor_obs: (*batch, D)  e.g. (512,) for eval, (B, 512) for training
                batch_prefix = actor_obs.shape[:-1]        # () for eval, (B,) for training
                actor_obs_expanded = jnp.repeat(actor_obs[..., None, :], N, axis=-2)
                # actor_obs_expanded: (*batch, N, D)
                noises = jax.random.normal(rng, (*batch_prefix, N, action_dim))
                actions = self.compute_flow_actions(actor_obs_expanded, noises, pre_encoded=True)
                actions = jnp.clip(actions, -1, 1)  # (*batch, N, action_dim)

                # Critic scoring: tile raw images so the critic's own CNN encoder is run
                # independently per (image, action) pair.  Works for both eval (no batch
                # dim) and training (batch dim present).
                n_batch_dims = len(batch_prefix)
                ob_inner = observations.shape[n_batch_dims:]   # (H, W, C)
                # Insert N-axis right after batch dims: (*batch, 1, H, W, C) → (*batch, N, H, W, C)
                obs_with_n = jnp.expand_dims(observations, n_batch_dims)
                obs_tiled = jnp.broadcast_to(obs_with_n, batch_prefix + (N,) + ob_inner)
                flat_obs = obs_tiled.reshape((-1,) + ob_inner)              # (prod(batch)*N, H, W, C)
                flat_actions = actions.reshape(-1, action_dim)
                q_flat = self.network.select("critic")(flat_obs, flat_actions)
                # q_flat: (num_qs, prod(batch)*N)
                q = q_flat.reshape((self.config['num_qs'],) + batch_prefix + (N,))
                # q: (num_qs, *batch, N)
                if self.config["q_agg"] == "mean":
                    q = q.mean(axis=0)   # (*batch, N)
                else:
                    q = q.min(axis=0)
            else:
                # 1-D latent obs: repeat raw latent for both actor and critic.
                # MLP encoders inside both networks handle arbitrary batch shapes.
                noises = jax.random.normal(
                    rng, (*observations.shape[:-1], N, action_dim)
                )
                obs_expanded = jnp.repeat(observations[..., None, :], N, axis=-2)  # (B, N, D)
                actions = self.compute_flow_actions(obs_expanded, noises, pre_encoded=False)
                actions = jnp.clip(actions, -1, 1)  # (B, N, action_dim)
                if self.config["q_agg"] == "mean":
                    q = self.network.select("critic")(obs_expanded, actions).mean(axis=0)
                else:
                    q = self.network.select("critic")(obs_expanded, actions).min(axis=0)

            indices = jnp.argmax(q, axis=-1)
            bshape = indices.shape
            indices = indices.reshape(-1)
            bsize = len(indices)
            actions = jnp.reshape(actions, (-1, N, action_dim))[
                jnp.arange(bsize), indices, :
            ].reshape(bshape + (action_dim,))

        return actions

    @functools.partial(jax.jit, static_argnames=('pre_encoded',))
    def compute_flow_actions(
        self,
        observations,
        noises,
        pre_encoded=False,
    ):
        """Compute actions from the BC flow model using the Euler method.

        Args:
            pre_encoded: If True, observations are already encoded (skip encoder).
                         Used when best-of-n pre-encodes image observations so the
                         repeat operates on 1D latents rather than raw images.
        """
        if not pre_encoded and self.config['encoder'] is not None:
            observations = self.network.select('actor_bc_flow_encoder')(observations)
        actions = noises
        # Euler method.
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_bc_flow')(observations, actions, t, is_encoded=True)
            actions = actions + vels / self.config['flow_steps']
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
        z_goal=None,
        wm_train_state=None,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
            wm_model: Optional Flax module (LeJEPAJaxForward) for differentiable
                rollouts. If provided alongside wm_params and z_goal, and
                config["rollout_loss_weight"] > 0, the actor loss includes
                an analytic-policy-gradient term.
            wm_params: Frozen Flax params for the WM.
            z_goal: (LATENT_DIM,) goal latent used to compute dense reward
                during rollouts.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]
        if config["action_chunking"]:
            full_actions = jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
        else:
            full_actions = ex_actions
        full_action_dim = full_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()
            encoders['actor_onestep_flow'] = encoder_module()

        # Define networks.
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_qs'],
            encoder=encoders.get('critic'),
        )

        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
            use_fourier_features=config["use_fourier_features"],
            fourier_feature_dim=config["fourier_feature_dim"],
        )
        actor_onestep_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_onestep_flow'),
        )

        
        network_info = dict(
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, full_actions, ex_times)),
            actor_onestep_flow=(actor_onestep_flow_def, (ex_observations, full_actions)),
            critic=(critic_def, (ex_observations, full_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
        )
        if encoders.get('actor_bc_flow') is not None:
            # Add actor_bc_flow_encoder to ModuleDict to make it separately callable.
            # NOTE: do NOT register critic's encoder here — using the same Python object
            # in two scopes causes Flax to share params, which empties out the encoder
            # params inside the target_critic scope.  The critic encoder is instead
            # invoked through the Value module directly via the reshape trick in
            # sample_actions (best-of-n / image encoder path).
            network_info['actor_bc_flow_encoder'] = (encoders.get('actor_bc_flow'), (ex_observations,))
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        if config["weight_decay"] > 0.:
            network_tx = optax.adamw(learning_rate=config['lr'], weight_decay=config["weight_decay"])
        else:
            network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params

        params[f'modules_target_critic'] = params[f'modules_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim

        return cls(
            rng,
            network=network,
            config=flax.core.FrozenDict(**config),
            wm_model=wm_model,
            wm_params=wm_params,
            z_goal=z_goal,
            wm_state=wm_train_state,
        )


def get_config():

    config = ml_collections.ConfigDict(
        dict(
            agent_name='acfql',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            q_agg='mean',  # Aggregation method for target Q values.
            alpha=100.0,  # BC coefficient (need to be tuned for each environment).
            num_qs=2, # critic ensemble size
            flow_steps=10,  # Number of flow steps.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            horizon_length=ml_collections.config_dict.placeholder(int), # will be set
            action_chunking=True,  # False means n-step return
            actor_type="distill-ddpg",
            actor_num_samples=32,  # for actor_type="best-of-n" only
            use_fourier_features=False,
            fourier_feature_dim=64,
            weight_decay=0.,
            # Differentiable-rollout (analytic policy gradient through frozen WM).
            # Disabled by default. Activates when rollout_loss_weight > 0 AND
            # wm_model / wm_params / z_goal are supplied at create() time.
            rollout_loss_weight=0.0,
            rollout_horizon=3,
            rollout_dense_scale=10.0,
            # Joint WM + policy training (offline phase). Enabled when the
            # agent is created with a non-None wm_train_state.
            joint_alpha=1.0,           # weight on L_pred (one-step prediction MSE)
            joint_beta=0.1,            # final weight on L_value (Bellman residual via WM)
            joint_beta_ramp_steps=50000,  # linear ramp from 0 to joint_beta
        )
    )
    return config
