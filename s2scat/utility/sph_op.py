from jax import jit, vmap
import jax.numpy as jnp
from functools import partial
from typing import List, Tuple
import s2fft

import s2wav
from s2wav.transforms import wavelet_precompute as wavelets

from s2fft.precompute_transforms.spherical import forward_transform_jax


@partial(jit)
def normalize_map(f: jnp.ndarray) -> jnp.ndarray:
    """Normalize the map I: mean=0 and std=1."""
    f -= jnp.nanmean(f)
    f /= jnp.nanstd(f)
    return f


@partial(jit, static_argnums=(1))
def make_flm_full(flm_real_only: jnp.ndarray, L: int) -> jnp.ndarray:
    """Some docstrings"""
    # Create and store signs
    msigns = (-1) ** jnp.arange(1, L)

    # Reflect and apply hermitian symmetry
    flm = jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128)
    flm = flm.at[:, L - 1 :].set(flm_real_only)
    flm = flm.at[:, : L - 1].set(jnp.flip(jnp.conj(flm[:, L:]) * msigns, axis=-1))

    return flm


@partial(jit, static_argnums=(0, 1, 2))
def quadrature(L: int, J_min: int, J_max: int):
    """Some docstrings"""
    J = s2wav.samples.j_max(L)
    quads = []
    for j in range(J_min, J_max + 1):
        Lj = s2wav.samples.wav_j_bandlimit(L, j, 2.0, True)
        quads.append(s2fft.utils.quadrature_jax.quad_weights(Lj, "mw"))
    return quads


@partial(jit, static_argnums=(1, 2, 3, 4))
def _first_flm_to_analysis(
    flm: jnp.ndarray,
    L: int,
    N: int,
    J_min: int,
    reality: bool,
    filters: Tuple[jnp.ndarray] = None,
    precomps: List[List[jnp.ndarray]] = None,
) -> jnp.ndarray:
    """Some docstrings"""
    return wavelets.flm_to_analysis(
        flm=flm,
        L=L,
        N=N,
        J_min=J_min,
        sampling="mw",
        reality=reality,
        filters=filters,
        precomps=precomps,
    )


@partial(jit, static_argnums=(1, 2, 3, 4))
def _forward_harmonic_vect(
    f: jnp.ndarray,
    j: int,
    Lj: int,
    J_min: int,
    reality: bool,
    precomps: List[List[jnp.ndarray]],
) -> jnp.ndarray:
    """Some docstrings"""
    vect_func = vmap(
        forward_transform_jax, in_axes=(0, None, None, None, None, None, None)
    )
    return vect_func(f, precomps[0][j - J_min], Lj, "mw", reality, 0, None)


@partial(jit, static_argnums=(1, 2, 3, 4, 5, 6, 7))
def _flm_to_analysis_vect(
    flmn: jnp.ndarray,
    j: int,
    Lj: int,
    L: int,
    N: int = 1,
    J_min: int = 0,
    J_max: int = None,
    reality: bool = False,
    filters: Tuple[jnp.ndarray] = None,
    precomps: List[List[jnp.ndarray]] = None,
) -> Tuple[jnp.ndarray]:
    """Some docstrings"""
    vect_func = vmap(
        wavelets.flm_to_analysis,
        in_axes=(0, None, None, None, None, None, None, None, None, None, None, None),
    )
    return vect_func(
        flmn,
        Lj,
        N,
        J_min,
        J_max,
        2.0,
        "mw",
        None,
        reality,
        filters[:, :Lj, L - Lj : L - 1 + Lj],
        [0, 0, precomps[2][: (j - 1) - J_min + 1]],
        False,
    )
