import functools
from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp

from utils.networks import MLP


class ResnetStack(nn.Module):
    """ResNet stack module."""

    num_features: int
    num_blocks: int
    max_pooling: bool = True

    @nn.compact
    def __call__(self, x):
        initializer = nn.initializers.xavier_uniform()
        conv_out = nn.Conv(
            features=self.num_features,
            kernel_size=(3, 3),
            strides=1,
            kernel_init=initializer,
            padding='SAME',
        )(x)

        if self.max_pooling:
            conv_out = nn.max_pool(
                conv_out,
                window_shape=(3, 3),
                padding='SAME',
                strides=(2, 2),
            )

        for _ in range(self.num_blocks):
            block_input = conv_out
            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.num_features,
                kernel_size=(3, 3),
                strides=1,
                padding='SAME',
                kernel_init=initializer,
            )(conv_out)

            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.num_features,
                kernel_size=(3, 3),
                strides=1,
                padding='SAME',
                kernel_init=initializer,
            )(conv_out)
            conv_out += block_input

        return conv_out


class ImpalaEncoder(nn.Module):
    """IMPALA encoder."""

    width: int = 1
    stack_sizes: tuple = (16, 32, 32)
    num_blocks: int = 2
    dropout_rate: float = None
    mlp_hidden_dims: Sequence[int] = (512,)
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x, train=True, cond_var=None):
        x = x.astype(jnp.float32) / 255.0

        conv_out = x

        for i, num_features in enumerate(self.stack_sizes):
            conv_out = ResnetStack(
                num_features=num_features * self.width,
                num_blocks=self.num_blocks,
                name=f'stack_{i}',
            )(conv_out)
            if self.dropout_rate is not None:
                conv_out = nn.Dropout(rate=self.dropout_rate, name=f'dropout_{i}')(
                    conv_out, deterministic=not train
                )

        conv_out = nn.relu(conv_out)
        if self.layer_norm:
            conv_out = nn.LayerNorm()(conv_out)
        out = conv_out.reshape((*x.shape[:-3], -1))

        out = MLP(self.mlp_hidden_dims, activate_final=True, layer_norm=self.layer_norm)(out)

        return out


class JEPAAdaptHead(nn.Module):
    """Trainable MLP head on top of frozen 192-D JEPA latents.

    Gives qc's actor/critic a learned projection from the frozen JEPA representation
    into a task-adapted space. Matches IMPALA's 512-D output so the same
    actor/critic hidden dims work for both B1 and B2.
    """

    hidden_dims: Sequence[int] = (512, 512)
    layer_norm: bool = True

    @nn.compact
    def __call__(self, x, train=True, cond_var=None):
        return MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)(x)


encoder_modules = {
    'impala': ImpalaEncoder,
    'impala_debug': functools.partial(ImpalaEncoder, num_blocks=1, stack_sizes=(4, 4)),
    'impala_small': functools.partial(ImpalaEncoder, num_blocks=1),
    'impala_large': functools.partial(ImpalaEncoder, stack_sizes=(64, 128, 128), mlp_hidden_dims=(1024,)),
    'jepa_head': JEPAAdaptHead,
    # Bigger 4-layer head: gives the trainable adapter more capacity to extract
    # task-relevant features from the frozen 192-D JEPA representation.
    # ~1.85M params per head (vs ~360K for default jepa_head).
    'jepa_head_deep': functools.partial(JEPAAdaptHead, hidden_dims=(1024, 1024, 512, 512)),
}
