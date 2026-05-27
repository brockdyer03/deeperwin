# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Curvature blocks for FermiNet.

Updated for newer kfac-jax versions where custom layer tags are represented
using the shared layer_tag primitive plus LayerMetaData(variant=...).
"""

from collections.abc import Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import kfac_jax
import numpy as np


Array = kfac_jax.utils.Array
Numeric = kfac_jax.utils.Numeric

vmap_matmul = jax.vmap(jnp.matmul, in_axes=(0, 0), out_axes=0)

# Newer kfac-jax uses one shared primitive for layer tags.
# It is exposed through the layers_and_loss_tags module, not as a separately
# constructed LayerTag("...", ...).
_LAYER_TAG = kfac_jax.layers_and_loss_tags.layer_tag


def _repeated_dense_metadata(num_params: int) -> kfac_jax.LayerMetaData:
    """Metadata telling kfac-jax how to split tag operands.

    The bound operands are ordered as:
      (output, input, weight[, bias])
    """
    return kfac_jax.LayerMetaData(
        variant="repeated_dense",
        outputs_index=(0,),
        inputs_index=(1,),
        params_index=tuple(range(2, 2 + num_params)),
    )


def register_repeated_dense(
    y: Array,
    x: Array,
    w: Array,
    b: Array | None = None,
    **kwargs: Any,
) -> Array:
    """Registers a repeated dense layer: y = matmul(x, w) + b.

    Use this around the output of repeated/vmapped dense computations when
    manually tagging the layer.
    """
    args = (y, x, w) if b is None else (y, x, w, b)
    return _LAYER_TAG.bind(
        *args,
        meta=_repeated_dense_metadata(num_params=len(args) - 2),
        **kwargs,
    )


class RepeatedDenseBlock(kfac_jax.DenseTwoKroneckerFactored):
    """Dense block repeatedly applied over leading input dimensions."""

    @property
    def scale(self) -> Numeric:
        """Number of repeated applications per example.

        For input shape [batch, repeat_1, ..., repeat_n, features], this returns
        repeat_1 * ... * repeat_n.
        """
        (x_shape,) = self.inputs_shapes
        return float(
            kfac_jax.utils.product(x_shape) // (x_shape[0] * x_shape[-1])
        )

    def update_curvature_matrix_estimate(
        self,
        state: Any,
        estimation_data: Mapping[str, Sequence[Array]],
        ema_old: Numeric,
        ema_new: Numeric,
        batch_size: int,
        pmap_axis_name: str | None = None,
    ) -> Any:
        """Flattens repeated leading dimensions before using dense K-FAC stats."""
        estimation_data = dict(**estimation_data)

        (x,) = estimation_data["inputs"]
        (dy,) = estimation_data["outputs_tangent"]

        assert x.shape[0] == batch_size

        estimation_data["inputs"] = (x.reshape((-1, x.shape[-1])),)
        estimation_data["outputs_tangent"] = (dy.reshape((-1, dy.shape[-1])),)

        flattened_batch_size = x.size // x.shape[-1]

        return super().update_curvature_matrix_estimate(
            state,
            estimation_data,
            ema_old,
            ema_new,
            flattened_batch_size,
            pmap_axis_name,
        )


def _dense(x: Array, params: Sequence[Array]) -> Array:
    """Example dense layer function used for graph pattern tracing."""
    w, *opt_b = params
    y = jnp.matmul(x, w)
    return y if not opt_b else y + opt_b[0]


def _dense_parameter_extractor(num_params: int):
    """Builds a graph-pattern parameter extractor for dense matmul patterns."""

    def extractor(eqns: Sequence[Any]) -> Mapping[str, Any]:
        for eqn in eqns:
            if eqn.primitive.name == "dot_general":
                return dict(
                    meta=_repeated_dense_metadata(num_params=num_params),
                    **eqn.params,
                )
        raise ValueError("Could not find dot_general in repeated dense pattern.")

    return extractor


# Build graph patterns for dense layers repeated under up to n_repeated_max vmaps.
repeated_dense_patterns = []

_dense_func = _dense
_dense_func_no_bias = _dense

_example_args_x = np.zeros([11, 13])
_example_args_w = np.zeros([13, 7])
_example_args_b = np.zeros([7])

n_repeated_max = 4

for n_rep in range(1, n_repeated_max + 1):
    _dense_func = jax.vmap(_dense_func, in_axes=(0, [None, None]))
    _dense_func_no_bias = jax.vmap(_dense_func_no_bias, in_axes=(0, [None]))

    # Only rank matters for graph matching. Avoid shape-1 dimensions because
    # they can trigger special-cased reshape/squeeze/broadcast behavior.
    _example_args_x = np.zeros([11 + n_rep, *_example_args_x.shape])

    pattern_dense = kfac_jax.tag_graph_matcher.GraphPattern(
        name=f"repeated_dense{n_rep}_with_bias",
        tag_primitive=_LAYER_TAG,
        compute_func=_dense_func,
        parameters_extractor_func=_dense_parameter_extractor(num_params=2),
        example_args=[_example_args_x, [_example_args_w, _example_args_b]],
    )

    pattern_dense_no_bias = kfac_jax.tag_graph_matcher.GraphPattern(
        name=f"repeated_dense{n_rep}_no_bias",
        tag_primitive=_LAYER_TAG,
        compute_func=_dense_func_no_bias,
        parameters_extractor_func=_dense_parameter_extractor(num_params=1),
        example_args=[_example_args_x, [_example_args_w]],
    )

    repeated_dense_patterns.append(pattern_dense)
    repeated_dense_patterns.append(pattern_dense_no_bias)


# Avoid duplicate pattern names if your installed kfac-jax already ships
# repeated_dense patterns in DEFAULT_GRAPH_PATTERNS.
_DEFAULT_PATTERN_NAMES = {p.name for p in kfac_jax.tag_graph_matcher.DEFAULT_GRAPH_PATTERNS}

GRAPH_PATTERNS = (
    tuple(
        p for p in repeated_dense_patterns
        if p.name not in _DEFAULT_PATTERN_NAMES
    )
    + kfac_jax.tag_graph_matcher.DEFAULT_GRAPH_PATTERNS
)


# The key is now the LayerMetaData.variant, not the old primitive name
# "repeated_dense_tag".
kfac_jax.set_default_tag_to_block_ctor(
    "repeated_dense",
    RepeatedDenseBlock,
)