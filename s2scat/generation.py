import jax
import jax.numpy as jnp
from functools import partial
from typing import List, Callable

import s2scat
from s2scat.operators import spherical
from s2scat import compression


def build_model(
    xlm: jnp.ndarray,
    L: int,
    N: int,
    J_min: int = 0,
    reality: bool = True,
    recursive: bool = False,
    isotropic: bool = False,
    delta_j: int = None,
    c_backend: bool = False,
) -> Callable:

    # Compute and cache wavelet matrices
    sigma = jnp.std(jnp.abs(xlm)[xlm != 0])
    config = s2scat.configure(L, N, J_min, reality, recursive, c_backend)

    # Compute normalisation
    norm = s2scat.compute_norm(xlm, L, N, J_min, reality, config, recursive)

    # Compute target statistics
    covariances = s2scat.scatter_c if c_backend else s2scat.scatter
    targets = covariances(
        xlm, L, N, J_min, reality, config, norm, recursive, isotropic, delta_j
    )

    # Define L2 loss function
    def loss_function(xlm):
        predicts = covariances(
            xlm, L, N, J_min, reality, config, norm, recursive, isotropic, delta_j
        )
        return s2scat.optimisation.l2_covariance_loss(predicts, targets)

    # Define single sampler
    def sampler(
        xlm: jnp.ndarray, niter: int, learning_rate: float
    ) -> List[jnp.ndarray]:
        xlm, history = s2scat.optimisation.fit_optax(
            xlm[0], loss_function, niter, learning_rate
        )
        xlm = s2scat.operators.spherical.make_flm_full(xlm, L) if reality else xlm
        return xlm, history

    # Define generative model
    def model(
        key: jnp.ndarray, count: int, niter: int, learning_rate: float
    ) -> List[jnp.ndarray]:
        xlm = _initial_arrays(key, sigma, L, count, reality)
        batched_sampler = jax.vmap(sampler, in_axes=(0, None, None))
        return batched_sampler(xlm, niter, learning_rate)

    return model


@partial(jax.jit, static_argnums=(2, 4))
def _initial_arrays(
    key: jnp.ndarray, sigma: float, L: int, count: int, reality: bool
) -> jnp.ndarray:
    keys = jax.random.split(key, count)
    func = jax.vmap(_random_array, in_axes=(0, None, None))
    return func(keys, L, reality) * sigma


@partial(jax.jit, static_argnums=(1, 2))
def _random_array(key: jnp.ndarray, L: int, reality: bool) -> jnp.ndarray:
    keys = jax.random.split(key, 2)
    xlm = jax.random.normal(
        keys[0], (L, L), dtype=jnp.float64
    ) + 1j * jax.random.normal(keys[1], (L, L), dtype=jnp.float64)
    return xlm if reality else s2scat.operators.spherical.make_flm_full(xlm, L)
