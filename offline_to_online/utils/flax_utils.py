import functools
import glob
import os
import pickle
from typing import Any, Dict, Mapping, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)


class ModuleDict(nn.Module):
    """A dictionary of modules.

    This allows sharing parameters between modules and provides a convenient way to access them.

    Attributes:
        modules: Dictionary of modules.
    """

    modules: Dict[str, nn.Module]

    @nn.compact
    def __call__(self, *args, name=None, **kwargs):
        """Forward pass.

        For initialization, call with `name=None` and provide the arguments for each module in `kwargs`.
        Otherwise, call with `name=<module_name>` and provide the arguments for that module.
        """
        if name is None:
            if kwargs.keys() != self.modules.keys():
                raise ValueError(
                    f'When `name` is not specified, kwargs must contain the arguments for each module. '
                    f'Got kwargs keys {kwargs.keys()} but module keys {self.modules.keys()}'
                )
            out = {}
            for key, value in kwargs.items():
                if isinstance(value, Mapping):
                    out[key] = self.modules[key](**value)
                elif isinstance(value, Sequence):
                    out[key] = self.modules[key](*value)
                else:
                    out[key] = self.modules[key](value)
            return out

        return self.modules[name](*args, **kwargs)


class TrainState(flax.struct.PyTreeNode):
    """Custom train state for models.

    Attributes:
        step: Counter to keep track of the training steps. It is incremented by 1 after each `apply_gradients` call.
        apply_fn: Apply function of the model.
        model_def: Model definition.
        params: Parameters of the model.
        tx: optax optimizer.
        opt_state: Optimizer state.
    """

    step: int
    apply_fn: Any = nonpytree_field()
    model_def: Any = nonpytree_field()
    params: Any
    tx: Any = nonpytree_field()
    opt_state: Any

    @classmethod
    def create(cls, model_def, params, tx=None, **kwargs):
        """Create a new train state."""
        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(
            step=1,
            apply_fn=model_def.apply,
            model_def=model_def,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )

    def __call__(self, *args, params=None, method=None, **kwargs):
        """Forward pass.

        When `params` is not provided, it uses the stored parameters.

        The typical use case is to set `params` to `None` when you want to *stop* the gradients, and to pass the current
        traced parameters when you want to flow the gradients. In other words, the default behavior is to stop the
        gradients, and you need to explicitly provide the parameters to flow the gradients.

        Args:
            *args: Arguments to pass to the model.
            params: Parameters to use for the forward pass. If `None`, it uses the stored parameters, without flowing
                the gradients.
            method: Method to call in the model. If `None`, it uses the default `apply` method.
            **kwargs: Keyword arguments to pass to the model.
        """
        if params is None:
            params = self.params
        variables = {'params': params}
        if method is not None:
            method_name = getattr(self.model_def, method)
        else:
            method_name = None

        return self.apply_fn(variables, *args, method=method_name, **kwargs)

    def select(self, name):
        """Helper function to select a module from a `ModuleDict`."""
        return functools.partial(self, name=name)

    def apply_gradients(self, grads, **kwargs):
        """Apply the gradients and return the updated state."""
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    def apply_loss_fn(self, loss_fn):
        """Apply the loss function and return the updated state and info.

        It additionally computes the gradient statistics and adds them to the dictionary.
        """
        grads, info = jax.grad(loss_fn, has_aux=True)(self.params)

        grad_max = jax.tree_util.tree_map(jnp.max, grads)
        grad_min = jax.tree_util.tree_map(jnp.min, grads)
        grad_norm = jax.tree_util.tree_map(jnp.linalg.norm, grads)

        grad_max_flat = jnp.concatenate([jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_max)], axis=0)
        grad_min_flat = jnp.concatenate([jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_min)], axis=0)
        grad_norm_flat = jnp.concatenate([jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_norm)], axis=0)

        final_grad_max = jnp.max(grad_max_flat)
        final_grad_min = jnp.min(grad_min_flat)
        final_grad_norm = jnp.linalg.norm(grad_norm_flat, ord=1)

        info.update(
            {
                'grad/max': final_grad_max,
                'grad/min': final_grad_min,
                'grad/norm': final_grad_norm,
            }
        )

        return self.apply_gradients(grads=grads), info


def save_agent(agent, save_dir, epoch):
    """Save the agent to a file.

    Args:
        agent: Agent.
        save_dir: Directory to save the agent.
        epoch: Epoch number.
    """

    save_dict = dict(
        agent=flax.serialization.to_state_dict(agent),
    )
    save_path = os.path.join(save_dir, f'params_{epoch}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(save_dict, f)

    print(f'Saved to {save_path}')


def restore_agent_with_file(agent, file_path):
    """Just like restore_agent() but expect file_path to include restore_epoch.

    Tolerates schema drift: any top-level keys in the saved state_dict that
    aren't present on the target agent are silently dropped (with a warning).
    This lets us load checkpoints saved by older code that had extra fields,
    e.g. when wm_params / z_goal were briefly pytree_node=True before being
    moved to nonpytree_field.
    """
    assert os.path.exists(file_path), f'File {file_path} does not exist'
    with open(file_path, 'rb') as f:
        load_dict = pickle.load(f)

    agent_state = load_dict['agent']
    # Drop unknown top-level fields (forward-compatible restore)
    target_sd = flax.serialization.to_state_dict(agent)
    target_keys = set(target_sd.keys())
    extra_keys = [k for k in agent_state.keys() if k not in target_keys]
    if extra_keys:
        print(f'  [restore] dropping unknown fields from ckpt: {extra_keys}')
        agent_state = {k: v for k, v in agent_state.items() if k in target_keys}

    # Replace fields with None when the target's slot is None.
    # (e.g., wm_state is a pytree field that defaults to None when joint
    # training is off; loading the raw dict into a None slot would yield a
    # dict that downstream code treats as a TrainState, breaking attribute
    # access). Flax requires the field to be present in the state_dict, so
    # we replace with None rather than removing.
    none_target_keys = [k for k in agent_state.keys()
                        if k in target_sd and target_sd[k] is None
                        and agent_state[k] is not None]
    if none_target_keys:
        print(f'  [restore] forcing ckpt fields to None where target slot is None: {none_target_keys}')
        agent_state = {**agent_state}
        for k in none_target_keys:
            agent_state[k] = None

    # Inject None for target slots that are None but absent from the source
    # state_dict (e.g., loading a pre-joint-training ckpt into the new agent
    # class that has wm_state=None). Flax requires every target field to be
    # present in the state_dict.
    missing_none_keys = [k for k in target_keys
                         if target_sd[k] is None and k not in agent_state]
    if missing_none_keys:
        print(f'  [restore] injecting None for target fields absent in ckpt: {missing_none_keys}')
        agent_state = {**agent_state}
        for k in missing_none_keys:
            agent_state[k] = None

    # Back-compat: critic_encoder was added as a separately-callable module
    # after some checkpoints were already saved.  If the target network expects
    # modules_critic_encoder but the checkpoint doesn't have it, bootstrap the
    # params from modules_actor_bc_flow_encoder (same architecture, same shape).
    # For 1-D latent encoders the critic_encoder path is never executed at eval
    # time (image_encoder=False), so the weights don't matter.  For image
    # encoders this gives a warm-start; new B1-BoN checkpoints will have the
    # key and won't hit this branch.
    ckpt_params = (agent_state.get('network') or {}).get('params') or {}
    tgt_params  = (target_sd.get('network')   or {}).get('params') or {}
    if ('modules_critic_encoder' in tgt_params and
            'modules_critic_encoder' not in ckpt_params and
            'modules_actor_bc_flow_encoder' in ckpt_params):
        print('  [restore] critic_encoder missing from ckpt — '
              'bootstrapping from actor_bc_flow_encoder params')
        agent_state = {**agent_state}
        agent_state['network'] = {**agent_state['network']}
        agent_state['network']['params'] = {**ckpt_params,
            'modules_critic_encoder': ckpt_params['modules_actor_bc_flow_encoder']}

    agent = flax.serialization.from_state_dict(agent, agent_state)

    print(f'Restored from {file_path}')

    return agent

def restore_agent(agent, restore_path, restore_epoch):
    """Restore the agent from a file.

    Args:
        agent: Agent.
        restore_path: Path to the directory containing the saved agent.
        restore_epoch: Epoch number.
    """
    candidates = glob.glob(restore_path)

    assert len(candidates) == 1, f'Found {len(candidates)} candidates: {candidates}'

    restore_path = candidates[0] + f'/params_{restore_epoch}.pkl'

    with open(restore_path, 'rb') as f:
        load_dict = pickle.load(f)

    agent = flax.serialization.from_state_dict(agent, load_dict['agent'])

    print(f'Restored from {restore_path}')

    return agent
