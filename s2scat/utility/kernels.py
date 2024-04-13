import jax.numpy as jnp
from typing import List
from s2wav import samples
from s2wav.transforms.construct import (
    generate_full_precomputes,
    generate_wigner_precomputes,
)
from s2fft.precompute_transforms.construct import spin_spherical_kernel_jax
from s2fft.recursions.price_mcewen import generate_precomputes_jax


def generate_precompute_matrices(
    L: int, N: int, J_min: int = 0, reality: bool = False
) -> List[List[jnp.ndarray]]:
    r"""Generates the full set of precompute matrices for the scattering transform with
        :math:`\mathcal{O}(NL^3)` memory overhead.

    Args:
        L (int): Spherical harmonic bandlimit.
        N (int): Azimuthal bandlimit (directionality).
        J_min(int, optional): Minimum dyadic wavelet scale to consider. Defaults to 0.
        reality (bool, optional): Whether :math:`f \in \mathbb{R}`, if True exploits
            hermitian symmetry of harmonic coefficients. Defaults to False.

    Returns:
        List[List[jnp.ndarray]]: First list is indexed by [spherical matrices, [], Wigner matrices],
            second list is indexed by wavelet scale :math:`j`. Within these nested lists
            one will find jnp.ndarrays with elements of the reduced Wigner-d matrices.

    Notes:
        Effectively, this uses the Price-McEwen recursion [1] to compute and cache the
        reduced Wigner-d matrices in a multiresolution manner, where each scale :math:`j`
        has its own set of matrices bandlimited at :math:`L_j, N_j`. This multiresolution
        approach means that the majority of the memory overhead is incurred for the highest
        wavelet scale :math:`J_{\text{max}}` alone.

        [1] Price, Matthew A., and Jason D. McEwen. "Differentiable and accelerated spherical
            harmonic and Wigner transforms." arXiv preprint arXiv:2311.14670 (2023).
    """
    J_max = samples.j_max(L)
    precomps = generate_full_precomputes(
        L=L, N=N, J_min=J_min, forward=False, reality=reality, nospherical=True
    )
    for j2 in range(J_min, J_max):
        Lj2 = samples.wav_j_bandlimit(L, j2, multiresolution=True)
        precomps[0].append(
            spin_spherical_kernel_jax(L=Lj2, spin=0, reality=reality, forward=True)
        )
    return precomps


def generate_recursive_matrices(
    L: int, N: int, J_min: int = 0, reality: bool = False
) -> List[jnp.ndarray]:
    r"""Generates a small set of recursive matrices for underlying Wigner-d recursion
        algorithms with a modest :math:`\mathcal{O}(NL^2)` memory overhead.

    Args:
        L (int): Spherical harmonic bandlimit.
        N (int): Azimuthal bandlimit (directionality).
        J_min(int, optional): Minimum dyadic wavelet scale to consider. Defaults to 0.
        reality (bool, optional): Whether :math:`f \in \mathbb{R}`, if True exploits
            hermitian symmetry of harmonic coefficients. Defaults to False.

    Returns:
        List[List[jnp.ndarray]]: First list is indexed by [spherical matrices, [], Wigner matrices],
            second list is indexed by wavelet scale :math:`j`. Within these nested lists
            one will find jnp.ndarrays with elements of the reduced Wigner-d matrices.

    Notes:
        These matrices are simply initial conditions and recursion update matrices used
        within the Price-McEwen Wigner-d recursion [1]. In principle these could be
        computed on-the-fly however computing and caching them a priori can improve
        the efficiency of the recursion.

        [1] Price, Matthew A., and Jason D. McEwen. "Differentiable and accelerated spherical
            harmonic and Wigner transforms." arXiv preprint arXiv:2311.14670 (2023).
    """
    J_max = samples.j_max(L)
    precomps = [[], [], []]
    precomps[2] = generate_wigner_precomputes(
        L, N, J_min=J_min, reality=reality, forward=False
    )
    for j2 in range(J_min, J_max):
        Lj2 = samples.wav_j_bandlimit(L, j2, multiresolution=True)
        precomps[0].append(generate_precomputes_jax(Lj2, forward=True))
    return precomps
