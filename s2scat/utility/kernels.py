import jax.numpy as jnp
from typing import List
from s2wav import samples
from s2wav.transforms.construct import generate_full_precomputes
from s2fft.precompute_transforms.construct import spin_spherical_kernel_jax


def generate_full_precompute(
    L: int,
    N: int,
    J_min: int = 0,
    lam: float = 2.0,
    sampling: str = "mw",
    nside: int = None,
    reality: bool = False,
) -> List[jnp.ndarray]:
    J_max = samples.j_max(L, lam=lam)
    precomps = generate_full_precomputes(
        L=L,
        N=N,
        J_min=J_min,
        lam=lam,
        forward=False,
        reality=reality,
        nospherical=True,
    )
    for j2 in range(J_min, J_max + 1):
        Lj2 = samples.wav_j_bandlimit(L, j2, lam, True)
        precomps[0].append(
            spin_spherical_kernel_jax(
                L=Lj2,
                spin=0,
                reality=reality,
                sampling=sampling,
                nside=nside,
                forward=True,
            )
        )
    return precomps
