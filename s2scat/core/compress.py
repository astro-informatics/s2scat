from jax import jit
import jax.numpy as jnp
from functools import partial
from typing import List, Tuple


@partial(jit, static_argnums=(2, 3))
def C01_C11_to_isotropic(
    C01: List[jnp.float64], C11: List[jnp.float64], J_min: int, J_max: int
) -> Tuple[jnp.ndarray]:
    r"""Convert the fourth (C01) and sixth (C11) order covariances to isotropic coefficients.

    Args:
        C01 (List[jnp.float64]): Fourth order covariance statistic :math:`\text{Cov}\big [ \Psi^{\lambda_1} f, \Psi^{\lambda_1} | \Psi^{\lambda_2} f | \big ]`.
        C11 (List[jnp.float64]): Sixth order covariance statistic :math:`\text{Cov}\big [ \Psi^{\lambda_1} | \Psi^{\lambda_3} f |, \Psi^{\lambda_1} | \Psi^{\lambda_2} f | \big ]`.
        J_min (int): Minimum dyadic wavelet scale to consider.
        J_max (int): Maximum dyadic wavelet scale to consider.

    Returns:
        Tuple[jnp.ndarray]: Isotropic fourth and sixth order scattering covariance statistics.

    Notes:
        For isotropic coefficients the statistics will be contracted across :math:`n`. This will
        dramatically compress the covariance representation, but will be somewhat less
        sensitive to directional structure.
    """
    for j1 in range(J_min, J_max):
        idx = j1 - J_min
        C01[idx] = jnp.einsum("ajn->a", C01[idx], optimize=True)
        C11[idx] = jnp.einsum("abjkn->ab", C11[idx], optimize=True)
    return C01, C11
