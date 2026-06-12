"""ReBRAC agent (vendored from ~/fql/agents/rebrac.py) for chunked-action latent offline RL.

Verbatim ReBRAC (TD3+BC with layer-norm and separate actor/critic BC penalties),
running on offline_to_online's identical Actor/Value/encoders. Treats the 25-D action
chunk as a flat action. Only change vs the original: sample_actions takes `rng` (API
parity with evaluate_real_ogbench) and returns the deterministic, noise-free mode for eval.

NOTE: get_config's alpha_actor/alpha_critic default to 0 (pure TD3 -> diverges offline);
the trainer sets nonzero BC coefficients. These are UNTUNED for this task.
"""
import copy
from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Actor, Value


class ReBRACAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        rng, sample_rng = jax.random.split(rng)
        next_dist = self.network.select('target_actor')(batch['next_observations'])
        next_actions = next_dist.mode()
        noise = jnp.clip(jax.random.normal(sample_rng, next_actions.shape) * self.config['actor_noise'],
                         -self.config['actor_noise_clip'], self.config['actor_noise_clip'])
        next_actions = jnp.clip(next_actions + noise, -1, 1)
        next_qs = self.network.select('target_critic')(batch['next_observations'], actions=next_actions)
        next_q = next_qs.min(axis=0)
        mse = jnp.square(next_actions - batch['next_actions']).sum(axis=-1)
        next_q = next_q - self.config['alpha_critic'] * mse
        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q
        q = self.network.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params)
        critic_loss = jnp.square(q - target_q).mean()
        return critic_loss, {'critic_loss': critic_loss, 'q_mean': q.mean()}

    def actor_loss(self, batch, grad_params, rng):
        dist = self.network.select('actor')(batch['observations'], params=grad_params)
        actions = dist.mode()
        qs = self.network.select('critic')(batch['observations'], actions=actions)
        q = jnp.min(qs, axis=0)
        mse = jnp.square(actions - batch['actions']).sum(axis=-1)
        lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
        actor_loss = -(lam * q).mean()
        bc_loss = (self.config['alpha_actor'] * mse).mean()
        total = actor_loss + bc_loss
        return total, {'total_loss': total, 'actor_loss': actor_loss, 'bc_loss': bc_loss, 'mse': mse.mean()}

    @partial(jax.jit, static_argnames=('full_update',))
    def total_loss(self, batch, grad_params, full_update=True, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)
        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v
        if full_update:
            actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
            for k, v in actor_info.items():
                info[f'actor/{k}'] = v
        else:
            actor_loss = 0.0
        return critic_loss + actor_loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'])
        network.params[f'modules_target_{module_name}'] = new_target_params

    @partial(jax.jit, static_argnames=('full_update',))
    def update(self, batch, full_update=True):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, full_update, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        if full_update:
            self.target_update(new_network, 'critic')
            self.target_update(new_network, 'actor')
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, rng=None, temperature=0.0):
        """Deterministic, noise-free eval action (TD3 mode)."""
        dist = self.network.select('actor')(observations, temperature=temperature)
        return jnp.clip(dist.mode(), -1, 1)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)
        action_dim = ex_actions.shape[-1]
        encoders = dict()
        if config['encoder'] is not None:
            enc = encoder_modules[config['encoder']]
            encoders['critic'] = enc(); encoders['actor'] = enc()
        critic_def = Value(hidden_dims=config['value_hidden_dims'], layer_norm=config['layer_norm'],
                           num_ensembles=2, encoder=encoders.get('critic'))
        actor_def = Actor(hidden_dims=config['actor_hidden_dims'], action_dim=action_dim,
                          layer_norm=config['actor_layer_norm'], tanh_squash=config['tanh_squash'],
                          state_dependent_std=False, const_std=True,
                          final_fc_init_scale=config['actor_fc_scale'], encoder=encoders.get('actor'))
        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor=(actor_def, (ex_observations,)),
            target_actor=(copy.deepcopy(actor_def), (ex_observations,)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)
        params = network.params
        params['modules_target_critic'] = params['modules_critic']
        params['modules_target_actor'] = params['modules_actor']
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    return ml_collections.ConfigDict(dict(
        agent_name='rebrac', lr=3e-4, batch_size=256,
        actor_hidden_dims=(512, 512, 512, 512), value_hidden_dims=(512, 512, 512, 512),
        layer_norm=True, actor_layer_norm=False, discount=0.99, tau=0.005,
        tanh_squash=True, actor_fc_scale=0.01,
        alpha_actor=0.0, alpha_critic=0.0, actor_freq=2,
        actor_noise=0.2, actor_noise_clip=0.5,
        encoder=ml_collections.config_dict.placeholder(str),
    ))
