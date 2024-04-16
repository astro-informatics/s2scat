from jax import jit, vmap
import jax.numpy as jnp
from functools import partial
from typing import List, Tuple
import s2fft

import s2wav
from s2wav.transforms import wavelet, wavelet_c, wavelet_precompute

from s2fft.precompute_transforms.spherical import forward_transform_jax
from s2fft.transforms.c_backend_spherical import ssht_forward
from s2fft.transforms.spherical import forward_jax


@partial(jit, static_argnums=(1))
def make_flm_full(flm_real_only: jnp.ndarray, L: int) -> jnp.ndarray:
    r"""Reflects real harmonic coefficients to a complete set of coefficients
        using hermitian symmetry.

    Args:
        flm_real_only (jnp.ndarray): Positive half-plane of the spherical harmonic coefficients.
        L (int): Spherical harmonic bandlimit.

    Returns:
        jnp.ndarray: Full set of spherical harmonic coefficients, reflected across :math:`m=0`
            by using hermitian symmetry.

    Notes:
        For real (spin-0) signals the harmonic coefficients obey :math:`f^*_{\ell, m} = (-1)^m f_{\ell, -m}`.
    """
    # Create and store signs
    msigns = (-1) ** jnp.arange(1, L)

    # Reflect and apply hermitian symmetry
    flm = jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128)
    flm = flm.at[:, L - 1 :].add(flm_real_only)
    flm = flm.at[:, : L - 1].add(jnp.flip(jnp.conj(flm[:, L:]) * msigns, axis=-1))

    return flm


@partial(jit, static_argnums=(1))
def make_flm_real(flm: jnp.ndarray, L: int) -> jnp.ndarray:
    r"""Compresses harmonic coefficients of a real signal into positive coefficients only
        which leverages hermitian symmetry.

    Args:
        flm (jnp.ndarray): Full set of spherical harmonic coefficients
        L (int): Spherical harmonic bandlimit.

    Returns:
        jnp.ndarray: Positive half-plane of the spherical harmonic coefficients

    Notes:
        For real (spin-0) signals the harmonic coefficients obey :math:`f^*_{\ell, m} = (-1)^m f_{\ell, -m}`.
    """
    return flm[:, L - 1 :]


@partial(jit, static_argnums=(0, 1))
def quadrature(L: int, J_min: int = 0) -> List[jnp.ndarray]:
    r"""Generates spherical quadrature weights associated Gauss-Legendre sampling points.

    Args:
        L (int): Spherical harmonic bandlimit.
        J_min(int, optional): Minimum dyadic wavelet scale to consider. Defaults to 0.

    Returns:
        List[jnp.ndarray]: Multiresolution quadrature weights for each :math:`\theta`
            corresponding to each wavelet scale :math:`j \in [J_{\text{min}}, J_{\text{max}}]`.
    """
    J_max = s2wav.samples.j_max(L)
    quads = []
    for j in range(J_min, J_max + 1):
        Lj = s2wav.samples.wav_j_bandlimit(L, j, multiresolution=True)
        quads.append(s2fft.utils.quadrature_jax.quad_weights(Lj, "gl"))
    return quads


@partial(jit, static_argnums=(1, 2, 3, 4, 5, 7))
def _forward_harmonic_vect(
    f: jnp.ndarray,
    j: int,
    Lj: int,
    J_min: int,
    J_max: int,
    reality: bool,
    precomps: List[List[jnp.ndarray]],
    recursive: bool = True,
) -> jnp.ndarray:
    """Private function which batches the forward SHT pass"""
    idx = j - J_min if j < J_max else j - J_min - 1
    return vmap(
        partial(
            forward_jax if recursive else forward_transform_jax,
            L=Lj,
            spin=0,
            nside=None,
            sampling="gl",
            reality=reality,
            **{"precomps" if recursive else "kernel": precomps[0][idx]},
        ),
        in_axes=0,
    )(f)


def _forward_harmonic_looped(f: jnp.ndarray, Lj: int, N: int) -> jnp.ndarray:
    """Private function for looped forward SHT pass (C bound functions)."""
    flm = jnp.zeros((2 * N - 1, Lj, 2 * Lj - 1), dtype=jnp.complex128)
    for n in range(2 * N - 1):
        flm = flm.at[n].add(ssht_forward(jnp.abs(f[n]), Lj, 0, True, 3))
    return flm


def _first_flm_to_analysis(
    flm: jnp.ndarray,
    L: int,
    N: int,
    J_min: int,
    reality: bool,
    filters: Tuple[jnp.ndarray] = None,
    precomps: List[List[jnp.ndarray]] = None,
    recursive: bool = True,
    use_c_backend: bool = False,
) -> jnp.ndarray:
    """Private function which interfaces with `s2wav.flm_to_analysis <https://astro-informatics.github.io/s2wav/api/transforms/wavelet.html#s2wav.transforms.wavelet.flm_to_analysis>`_."""
    if use_c_backend and not recursive:
        raise ValueError(
            "C backend functions do not support full precompute transform."
        )
    precomps_slice = precomps[2] if recursive and not use_c_backend else precomps
    args = {"precomps": precomps_slice} if not use_c_backend else {}
    submod = (
        wavelet_c if use_c_backend else (wavelet if recursive else wavelet_precompute)
    )
    return submod.flm_to_analysis(
        flm=flm,
        L=L,
        N=N,
        J_min=J_min,
        sampling="gl",
        reality=reality,
        filters=filters,
        **args,
    )


@partial(jit, static_argnums=(1, 2, 3, 4, 5, 6, 7, 10, 11))
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
    recursive: bool = True,
    delta_j: int = None,
) -> Tuple[jnp.ndarray]:
    """Private function with batches the partial analysis transform."""
    args = {} if recursive else {"_precomp_shift": False}
    idx = j - J_min
    j_min = J_min if delta_j is None else max(J_min, j - delta_j - 1)
    preslice = (
        precomps[2][j_min - J_min : idx]
        if recursive
        else [0, 0, precomps[2][j_min - J_min : idx]]
    )
    filtslice = filters[:, :Lj, L - Lj : L - 1 + Lj]
    submod = wavelet if recursive else wavelet_precompute
    return vmap(
        partial(
            submod.flm_to_analysis,
            L=Lj,
            N=N,
            J_min=j_min,
            J_max=J_max,
            sampling="gl",
            reality=reality,
            filters=filtslice,
            precomps=preslice,
            **args,
        ),
        in_axes=0,
    )(flmn)


def _flm_to_analysis_looped(
    flmn: jnp.ndarray,
    j: int,
    Lj: int,
    L: int,
    N: int = 1,
    J_min: int = 0,
    J_max: int = None,
    filters: Tuple[jnp.ndarray] = None,
    delta_j: int = None,
) -> Tuple[jnp.ndarray]:
    """Private function which loops over the partial analysis transform (C bound functions)."""
    j_min = J_min if delta_j is None else max(J_min, j - delta_j - 1)

    f_wav = [[] for _ in range(j_min, J_max + 1)]
    temp = filters[:, :Lj, L - Lj : L - 1 + Lj]
    for n in range(2 * N - 1):
        f_wav_n = wavelet_c.flm_to_analysis(
            flmn[n], Lj, N, j_min, J_max, 2.0, "gl", True, temp
        )
        for j in range(j_min, J_max + 1):
            f_wav[j - j_min].append(f_wav_n[j - j_min])

    return [jnp.array(f_wav[i]) for i in range(len(f_wav))]
