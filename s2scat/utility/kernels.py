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
) -> List[jnp.ndarray]:
    """some docstrings"""
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
    """some docstrings"""
    J_max = samples.j_max(L)
    precomps = [[], [], []]
    precomps[2] = generate_wigner_precomputes(
        L, N, J_min=J_min, reality=reality, forward=False
    )
    for j2 in range(J_min, J_max):
        Lj2 = samples.wav_j_bandlimit(L, j2, multiresolution=True)
        precomps[0].append(generate_precomputes_jax(Lj2, forward=True))
    return precomps
