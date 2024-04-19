from jax import jit
import jax.numpy as jnp
from functools import partial
from typing import List, Tuple


import s2wav
from s2scat.operators import spherical
from s2scat.utility.statistics import add_to_P00


@partial(jit, static_argnums=(1, 2, 3, 4, 7))
def compute_norm(
    flm: jnp.ndarray,
    L: int,
    N: int,
    J_min: int = 0,
    reality: bool = False,
    filters: jnp.ndarray = None,
    precomps: List[List[jnp.ndarray]] = None,
    recursive: bool = True,
) -> List[jnp.ndarray]:
    r"""Compute multi-scale normalisation for the scattering covariances.

    Args:
        flm (jnp.ndarray): Spherical harmonic coefficients.
        L (int): Spherical harmonic bandlimit.
        N (int): Azimuthal bandlimit (directionality).
        J_min (int, optional): Minimum dyadic wavelet scale to consider. Defaults to 0.
        reality (bool, optional): Whether :math:`f \in \mathbb{R}`, if True exploits
            hermitian symmetry of harmonic coefficients. Defaults to False.
        filters (jnp.ndarray, optional): Directional wavelet filters defined in spherical
            harmonic space. Defaults to None, but note that currently this is not supported.
        precomps (List[jnp.ndarray], optional): Various cached precomputed values. Defaults
            to None, but note that currently this is not supported.
        recursive (bool, optional): Whether to perform a memory efficient recursive transform,
            or a faster but less memory efficient fully precompute transform. Defaults to True.

    Raises:
        ValueError: If one does not pass an array of precomps.
        ValueError: If one does not pass an array of wavelet filters.

    Returns:
        Tuple[jnp.ndarray]: Normalisation for the scattering covariance statistics.
    """
    if precomps is None:
        raise ValueError("Must provide precomputed kernels for this transform!")

    if filters is None:
        raise ValueError("Must provide wavelet filters for this transform!")

    ### Configure maximum scale, impose reality, define quadrature
    J_max = s2wav.samples.j_max(L)
    Q = spherical.quadrature(L, J_min)
    flm = spherical.make_flm_full(flm, L) if reality else flm

    ### Perform first wavelet transform W_j2 = f * Psi_j2
    W = spherical._first_flm_to_analysis(
        flm, L, N, J_min, reality, filters, precomps, recursive
    )

    P00 = []
    for j2 in range(J_min, J_max + 1):
        idx = j2 - J_min
        P00 = add_to_P00(P00, W[idx], Q[idx])

    return P00


@partial(jit, static_argnums=(5, 6))
def apply_norm(
    S1: List[jnp.float64],
    P00: List[jnp.float64],
    C01: List[jnp.float64],
    C11: List[jnp.float64],
    norm: List[jnp.ndarray],
    J_min: int,
    J_max: int,
) -> Tuple[List[jnp.ndarray]]:
    r"""Applies multi-scale normalisation to the scattering covariances.

    Args:
        S1 (List[jnp.float64]): Mean field statistic :math:`\langle |\Psi^\lambda f| \rangle`.
        P00 (List[jnp.float64]): Second order power statistic :math:`\langle |\Psi^\lambda f|^2 \rangle`.
        C01 (List[jnp.float64]): Fourth order covariance statistic :math:`\text{Cov}\big [ \Psi^{\lambda_1} f, \Psi^{\lambda_1} | \Psi^{\lambda_2} f | \big ]`.
        C11 (List[jnp.float64]): Sixth order covariance statistic :math:`\text{Cov}\big [ \Psi^{\lambda_1} | \Psi^{\lambda_3} f |, \Psi^{\lambda_1} | \Psi^{\lambda_2} f | \big ]`.
        norm (List[jnp.ndarray]): Multiscale normalisation factors for given signal.
        J_min (int): Minimum dyadic wavelet scale to consider.
        J_max (int): Maximum dyadic wavelet scale to consider.

    Returns:
        Tuple[List[jnp.ndarray]]: Tuple of normalised scattering covariance statistics.
    """
    for j2 in range(J_min, J_max + 1):
        idx = j2 - J_min
        S1[idx] /= jnp.sqrt(norm[idx])
        P00[idx] /= norm[idx]

    for j1 in range(J_min, J_max):
        idx = j1 - J_min
        C01[idx] = jnp.einsum(
                "ajn,j->ajn",
                C01[idx],
                1 / jnp.sqrt(norm[idx]),
                optimize=True
            )
        C01[idx] = jnp.einsum(
            "ajn,n->ajn",
            C01[idx],
            1 / jnp.sqrt(norm[idx]),
            optimize=True
        )
        C11[idx] = jnp.einsum(
                "abjkn,j->abjkn",
                C11[idx],
                1 / jnp.sqrt(norm[idx]),
                optimize=True
            )
        C11[idx] = jnp.einsum(
            "abjkn,k->abjkn",
            C11[idx],
            1 / jnp.sqrt(norm[idx]),
            optimize=True
        )
    return S1, P00, C01, C11
