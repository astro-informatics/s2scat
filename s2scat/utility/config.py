from typing import Tuple, List
import jax.numpy as jnp
import s2scat
import s2wav


def run_config(
    L: int, N: int, J_min: int = 0, reality: bool = False, recursive: bool = True
) -> Tuple[List[jnp.ndarray]]:
    """Generates and caches all precomputed arrays e.g. quadrature and recursive updates.

    Args:
        L (int): Spherical harmonic bandlimit
        N (int): Azimuthal bandlimit (directionality)
        J_min (int, optional): Minimum wavelet scale. Defaults to 0.
        reality (bool, optional): Whether :math:`f \in \mathbb{R}`, if True exploits
            hermitian symmetry of harmonic coefficients. Defaults to False.
        recursive (bool, optional): Whether to perform a memory efficient recursive transform,
            or a faster but less memory efficient fully precompute transform. Defaults to True.

    Returns:
        Tuple[List[jnp.ndarray]]: All necessary precomputed arrays.
    """
    generator = (
        s2scat.operators.matrices.generate_recursive_matrices
        if recursive
        else s2scat.operators.matrices.generate_precompute_matrices
    )
    matrices = generator(L, N, J_min, reality)
    quadrature = s2scat.operators.spherical.quadrature(L, J_min)
    wavelets = s2wav.filters.filters_directional_vectorised(L, N)[0]
    return wavelets, quadrature, matrices
