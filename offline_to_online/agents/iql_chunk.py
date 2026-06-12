"""IQL agent (vendored from ~/fql/agents/iql.py) for chunked-action latent offline RL.

Verbatim standard IQL — expectile value, Q-bootstrap critic, AWR actor — operating
on the offline_to_online utils (same Actor/Value/ModuleDict/encoders, since this repo
derives from the same FQL codebase). The 25-D action chunk is treated as a flat action;
chunking only matters at execution (the eval dispatches 5 actions per 25-D output).

Only change vs the original: sample_actions takes `rng` (to match evaluate_real_ogbench's
`agent.sample_actions(observations=z, rng=key)` call) and returns the deterministic mode
(standard IQL/AWR eval).
"""
import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Actor, Value


class IQLAgent(flax.struct.PyTreeNode):
    """Implicit Q-learning (IQL) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def value_loss(self, batch, grad_params):
        q1, q2 = self.network.select('target_critic')(batch['observations'], actions=batch['actions'])
        q = jnp.minimum(q1, q2)
        v = self.network.select('value')(batch['observations'], params=grad_params)
        value_loss = self.expectile_loss(q - v, q - v, self.config['expectile']).mean()
        return value_loss, {'value_loss': value_loss, 'v_mean': v.mean()}

    def critic_loss(self, batch, grad_params):
        next_v = self.network.select('value')(batch['next_observations'])
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v
        q1, q2 = self.network.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params)
        critic_loss = ((q1 - q) ** 2 + (q2 - q) ** 2).mean()
        return critic_loss, {'critic_loss': critic_loss, 'q_mean': q.mean()}

    def actor_loss(self, batch, grad_params, rng=None):
        if self.config['actor_loss'] == 'awr':
            v = self.network.select('value')(batch['observations'])
            q1, q2 = self.network.select('critic')(batch['observations'], actions=batch['actions'])
            q = jnp.minimum(q1, q2)
            adv = q - v
            exp_a = jnp.minimum(jnp.exp(adv * self.config['alpha']), 100.0)
            dist = self.network.select('actor')(batch['observations'], params=grad_params)
            log_prob = dist.log_prob(batch['actions'])
            actor_loss = -(exp_a * log_prob).mean()
            return actor_loss, {
                'actor_loss': actor_loss, 'adv': adv.mean(),
                'bc_log_prob': log_prob.mean(),
                'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                'std': jnp.mean(dist.scale_diag),
            }
        elif self.config['actor_loss'] == 'ddpgbc':
            dist = self.network.select('actor')(batch['observations'], params=grad_params)
            if self.config['const_std']:
                q_actions = jnp.clip(dist.mode(), -1, 1)
            else:
                q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)
            q1, q2 = self.network.select('critic')(batch['observations'], actions=q_actions)
            q = jnp.minimum(q1, q2)
            q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean())
            log_prob = dist.log_prob(batch['actions'])
            bc_loss = -(self.config['alpha'] * log_prob).mean()
            actor_loss = q_loss + bc_loss
            return actor_loss, {'actor_loss': actor_loss, 'q_loss': q_loss, 'bc_loss': bc_loss}
        else:
            raise ValueError(f'Unsupported actor loss: {self.config["actor_loss"]}')

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v
        critic_loss, critic_info = self.critic_loss(batch, grad_params)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v
        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v
        loss = value_loss + critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, rng=None, temperature=0.0):
        """Deterministic eval action (AWR/IQL mode). rng accepted for API parity."""
        dist = self.network.select('actor')(observations, temperature=temperature)
        actions = jnp.clip(dist.mode(), -1, 1)
        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)
        action_dim = ex_actions.shape[-1]

        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['value'] = encoder_module()
            encoders['critic'] = encoder_module()
            encoders['actor'] = encoder_module()

        value_def = Value(hidden_dims=config['value_hidden_dims'], layer_norm=config['layer_norm'],
                          num_ensembles=1, encoder=encoders.get('value'))
        critic_def = Value(hidden_dims=config['value_hidden_dims'], layer_norm=config['layer_norm'],
                           num_ensembles=2, encoder=encoders.get('critic'))
        actor_def = Actor(hidden_dims=config['actor_hidden_dims'], action_dim=action_dim,
                          layer_norm=config['actor_layer_norm'], state_dependent_std=False,
                          const_std=config['const_std'], encoder=encoders.get('actor'))

        network_info = dict(
            value=(value_def, (ex_observations,)),
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor=(actor_def, (ex_observations,)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)
        params = network_params
        params['modules_target_critic'] = params['modules_critic']
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    return ml_collections.ConfigDict(dict(
        agent_name='iql', lr=3e-4, batch_size=256,
        actor_hidden_dims=(512, 512, 512, 512), value_hidden_dims=(512, 512, 512, 512),
        layer_norm=True, actor_layer_norm=False, discount=0.99, tau=0.005,
        expectile=0.9, actor_loss='awr', alpha=10.0, const_std=True,
        encoder=ml_collections.config_dict.placeholder(str),
    ))
