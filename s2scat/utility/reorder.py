from jax import jit
import jax.numpy as jnp
from functools import partial
from typing import List, Tuple


@partial(jit, static_argnums=(1, 2))
def nested_list_to_list_of_arrays(
    Nj1j2: List[List[List[jnp.ndarray]]], J_min: int, J_max: int
) -> List[jnp.ndarray]:
    r"""Specific reindexing function which switches covariance wavelet scale list ordering.

    Args:

        Nj1j2 (List[List[jnp.ndarray]]): Nested list of second order wavelet coefficients.
        J_min(int): Minimum dyadic wavelet scale to consider.
        J_max(int): Maximum dyadic wavelet scale to consider.

    Returns:
        List[jnp.ndarray]: Compressed list of all scattering covariances that end in scale :math:`j2`.

    Notes:
        Within the scattering covariance transform one performs the wavelet transform of
        the modulus of the wavelet transform of a signal. Consequently, one recovers nested
        lists indexed by the two wavelet scales :math:`j_1, j_2` for the two wavelet
        transforms. The harmonic bandlimit of the output signal is capped by the lower
        of these two scales, which by construction is the second (the modulus operation
        forces information to lower scales).

        Therefore, by reversing scale indexing :math:`j_1 \leftrightarrow j_2` all
        internal arrays are of the same dimension, which simplifies the subsequent
        covariance statistics which are computed from combindations of these arrays.
    """
    Nj1j2_flat = []
    for j1 in range(J_min, J_max):
        idx1 = j1 - J_min
        Nj1j2_flat_for_j2 = []
        for j2 in range(j1 + 1, J_max + 1):
            idx2 = j2 - J_min - 1
            Nj1j2_flat_for_j2.append(Nj1j2[idx2][idx1])
        Nj1j2_flat.append(jnp.array(Nj1j2_flat_for_j2))
    return Nj1j2_flat


@jit
def list_to_array(
    S1: List[jnp.float64],
    P00: List[jnp.float64],
    C01: List[jnp.float64],
    C11: List[jnp.float64],
) -> Tuple[jnp.ndarray]:
    r"""Converts list of statistics to array of statistics for e.g. Loss functions.

    Args:
        S1 (List[jnp.float64]): Mean field statistic :math:`\langle |\Psi^\lambda f| \rangle`.
        P00 (List[jnp.float64]): Second order power statistic :math:`\langle |\Psi^\lambda f|^2 \rangle`.
        C01 (List[jnp.float64]): Fourth order covariance statistic :math:`\text{Cov}\big [ \Psi^\lambda_1 f, \Psi^{\lambda_1} | \Psi^{\lambda_2} f | \big ]`.
        C11 (List[jnp.float64]): Sixth order covariance statistic :math:`\text{Cov}\big [ \Psi^{\lambda_1} | \Psi^{\lambda_3} f |, \Psi^{\lambda_1} | \Psi^{\lambda_2} f | \big ]`.

    Returns:
        Tuple[jnp.ndarray]: Tuple of (S1, P00, C01, C11) arrays encoding scattering covariance statistics.

    Notes:
        :math:`\lambda = (j, n)` is an index which corresponds to both scale :math:`j` and
        direction :math:`n`. Hence each :math:`\lambda` corresponds to two degrees of freedom making, for example,
        C11 a sixth order covariance.
    """
    S1 = jnp.concatenate(S1)
    P00 = jnp.concatenate(P00)
    C01 = jnp.concatenate(C01, axis=None)
    C11 = jnp.concatenate(C11, axis=None)
    return S1, P00, C01, C11
