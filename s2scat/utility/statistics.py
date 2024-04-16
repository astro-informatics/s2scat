from jax import jit
import jax.numpy as jnp
from functools import partial
from typing import List, Tuple


@partial(jit)
def normalize_map(f: jnp.ndarray) -> jnp.ndarray:
    r"""Normalises a spherical map by removing the mean and enforcing unit variance.

    Args:
        f (jnp.ndarray): Signal on the sphere.

    Returns:
        jnp.ndarray: Spherical signal with zero mean and unit variance.
    """
    f -= jnp.nanmean(f)
    f /= jnp.nanstd(f)
    return f


@partial(jit, static_argnums=(1))
def compute_mean_variance(flm: jnp.ndarray, L: int) -> Tuple[jnp.float64]:
    r"""Computes the mean and variance of a spherical signal.

    Args:
        flm (jnp.ndarray): Spherical harmonic coefficients.
        L (int): Spherical harmonic bandlimit.

    Returns:
        Tuple[jnp.float64]: mean and variance of provided signal in harmonic space.

    Notes:
        The mean of a spherical signal can easily be read off from the harmonic coefficients
        as the the mean is just the :math:`f_{00}` coefficient. Here we choose to capture the
        variance in harmonic space for simplicity.
    """
    mean = jnp.abs(flm[0, L - 1] / (2 * jnp.sqrt(jnp.pi)))
    Ilm_square = flm * jnp.conj(flm)
    var = (jnp.sum(Ilm_square) - Ilm_square[0, L - 1]) / (4 * jnp.pi)
    return mean, var


@partial(jit, static_argnums=(1, 2, 3, 4, 5))
def compute_P00(
    W: List[jnp.ndarray],
    Q: List[jnp.ndarray],
    N: List[jnp.ndarray],
    S: bool,
    J_min: int,
    J_max: int,
) -> jnp.ndarray:
    r"""Compute the second order power statistics.

    Args:
        W (List[jnp.ndarray]): Multiscale first order wavelet coefficients.
        Q (List[jnp.ndarray]): Multiscale quadrautre weights of given sampling pattern.
        N (List[jnp.ndarray]): Multiscale normalisation factors for given signal.
        S (bool): Whether to return statitic as an array or list.
        J_min (int): Minimum dyadic wavelet scale to consider.
        J_max (int): Maximum dyadic wavelet scale to consider.

    Returns:
        jnp.ndarray: Second order power statistics.

    Notes:
        These second order statistics :math:`P00=\langle |\Psi^\lambda f|^2 \rangle` can
        be seen as the average power of each wavelet scale :math:`j` and direction
        :math:`n` and as such help to ensure the covariances capture the power spectrum
        effectively.
    """
    P00 = []
    for j2 in range(J_min, J_max + 1):
        idx = j2 - J_min
        P00 = add_to_P00(P00, W[idx], Q[idx])

    if N is not None:
        for j2 in range(J_min, J_max + 1):
            idx = j2 - J_min
            P00[idx] /= N[idx]
    P00 = jnp.concatenate(P00) if S else P00
    return P00


@partial(jit, static_argnums=(3, 4, 5))
def compute_C01_and_C11(
    Nj1j2: List[jnp.ndarray],
    W: List[jnp.ndarray],
    Q: List[jnp.ndarray],
    J_min: int,
    J_max: int,
    isotropic: bool = False,
) -> Tuple[jnp.ndarray]:
    r"""Compute the fourth (C01) and sixth (C11) order covariances.

    Args:
        Nj1j2 (List[jnp.ndarray]): Multiscale second order wavelet coefficients.
        W (List[jnp.ndarray]): Multiscale first order wavelet coefficients.
        Q (List[jnp.ndarray]): Multiscale quadrautre weights of given sampling pattern.
        J_min (int): Minimum dyadic wavelet scale to consider.
        J_max (int): Maximum dyadic wavelet scale to consider.
        isotropic (bool, optional): Whether to return isotropic coefficients, i.e. average
            over directionality. Defaults to False.

    Returns:
        Tuple[jnp.ndarray]: Fourth and sixth order scattering covariance statistics.

    Notes:
        The fourth order statistic :math:`\text{C01} = \text{Cov}\big [ \Psi^{\lambda_1} f, \Psi^{\lambda_1} | \Psi^{\lambda_2} f | \big ]`
        can be seen as the covariance between power on scales :math:`\lambda_1=(j_1,n_1)`
        and :math:`\lambda_2=(j_2,n_2)`, and is hence a fourth order statistic which is
        sensitive to directional multiscale structure, e.g. filaments. C11 behaves similarly,
        but correlates between three scales and directions and is consequently sixth order.

        If isotropic is true, the statistics will be contracted across :math:`n`. This will
        dramatically compress the covariance representation, but will be somewhat less
        sensitive to directional structure.
    """
    C01 = []
    C11 = []
    for j1 in range(J_min, J_max):
        idx = j1 - J_min
        C01 = add_to_C01(C01, Nj1j2[idx], W[idx], Q[idx], isotropic)
        C11 = add_to_C11(C11, Nj1j2[idx], Q[idx], isotropic)
    return C01, C11


@partial(jit, static_argnums=(2))
def add_to_S1(S1: List[jnp.float64], Mlm: jnp.ndarray, L: int) -> List[jnp.float64]:
    r"""Computes the mean field statistic :math:`\text{S1}_j = \langle |\Psi^\lambda f| \rangle` at scale :math:`j`.

    Args:
        S1 (List[jnp.float64]): List in which to append the mean field statistic.
        Mlm (jnp.ndarray): Spherical harmonic coefficients, batched at index 0 over directionality.
        L (int): Spherical harmonic bandlimit.

    Returns:
        List[jnp.float64]: List into which :math:`\text{S1}_j` has been appended.
    """
    val = Mlm[:, 0, L - 1] / (2 * jnp.sqrt(jnp.pi))
    S1.append(jnp.real(val))
    return S1


@jit
def add_to_P00(
    P00: List[jnp.float64], W: jnp.ndarray, Q: jnp.ndarray
) -> List[jnp.float64]:
    r"""Computes the second order power statistic :math:`\text{P00}_j = \langle |\Psi^\lambda f|^2 \rangle` at scale :math:`j`.

    Args:
        P00 (List[jnp.float64]): List in which to append the second order power statistic.
        W (jnp.ndarray): Spherical signal at a single scale :math:`j`.
        Q (List[jnp.ndarray]): Quadrautre weights of given sampling pattern at scale :math:`j`.

    Returns:
        List[jnp.float64]: List into which :math:`\text{P00}_j` has been appended.
    """
    val = jnp.sum((jnp.abs(W) ** 2) * Q[None, :, None], axis=(-1, -2)) / (4 * jnp.pi)
    P00.append(jnp.real(val))
    return P00


@partial(jit, static_argnums=(4))
def add_to_C01(
    C01: List[jnp.float64],
    Nj1j2: jnp.ndarray,
    W: jnp.ndarray,
    Q: jnp.ndarray,
    isotropic: bool = False,
) -> List[jnp.float64]:
    r"""Computes the fourth order covariance statistic :math:`\text{C01}_j = \text{Cov}\big [ \Psi^{\lambda_1} f, \Psi^{\lambda_1} | \Psi^{\lambda_2} f | \big ]` at scale :math:`j`.

    Args:
        C01 (List[jnp.float64]): List in which to append the fourth order covariance statistic.
        Nj1j2 (List[jnp.ndarray]): Second order wavelet coefficients at scale :math:`j`.
        W (jnp.ndarray): Spherical signal at a single scale :math:`j`.
        Q (List[jnp.ndarray]): Quadrautre weights of given sampling pattern at scale :math:`j`.
        isotropic (bool, optional): Whether to return isotropic coefficients, i.e. average
            over directionality. Defaults to False.

    Returns:
        List[jnp.float64]: List into which :math:`\text{C01}_j` has been appended.

    Notes:
        If isotropic is true, the statistics will be contracted across :math:`n`. This will
        dramatically compress the covariance representation, but will be somewhat less
        sensitive to directional structure.
    """
    einsum_str = "ajntp,ntp,t->a" if isotropic else "ajntp,ntp,t->ajn"
    val = jnp.einsum(einsum_str, jnp.conj(Nj1j2), W, Q, optimize=True)
    C01.append(jnp.real(val))
    return C01


@partial(jit, static_argnums=(3))
def add_to_C11(
    C11: List[jnp.float64], Nj1j2: jnp.ndarray, Q: jnp.ndarray, isotropic: bool = False
) -> List[jnp.float64]:
    r"""Computes the sixth order covariance statistic :math:`\text{C11}_j = \text{Cov}\big [ \Psi^{\lambda_1} | \Psi^{\lambda_3} f |, \Psi^{\lambda_1} | \Psi^{\lambda_2} f | \big ]` at scale :math:`j`.

    Args:
        C11 (List[jnp.float64]): List in which to append the sixth order covariance statistic.
        Nj1j2 (List[jnp.ndarray]): Second order wavelet coefficients at scale :math:`j`.
        Q (List[jnp.ndarray]): Quadrautre weights of given sampling pattern at scale :math:`j`.
        isotropic (bool, optional): Whether to return isotropic coefficients, i.e. average
            over directionality. Defaults to False.

    Returns:
        List[jnp.float64]: List into which :math:`\text{C11}_j` has been appended.

    Notes:
        If isotropic is true, the statistics will be contracted across :math:`n`. This will
        dramatically compress the covariance representation, but will be somewhat less
        sensitive to directional structure.
    """
    einsum_str = "ajntp,bkntp, t->ab" if isotropic else "ajntp,bkntp, t->abjkn"
    val = jnp.einsum(einsum_str, Nj1j2, jnp.conj(Nj1j2), Q, optimize=True)
    C11.append(jnp.real(val))
    return C11


@partial(jit, static_argnums=(5, 6))
def apply_norm(
    S1: List[jnp.float64],
    P00: List[jnp.float64],
    C01: List[jnp.float64],
    C11: List[jnp.float64],
    N: List[jnp.ndarray],
    J_min: int,
    J_max: int,
) -> Tuple[List[jnp.ndarray]]:
    r"""Applies normalisation to a list of statistics.

    Args:
        S1 (List[jnp.float64]): Mean field statistic :math:`\langle |\Psi^\lambda f| \rangle`.
        P00 (List[jnp.float64]): Second order power statistic :math:`\langle |\Psi^\lambda f|^2 \rangle`.
        C01 (List[jnp.float64]): Fourth order covariance statistic :math:`\text{Cov}\big [ \Psi^{\lambda_1} f, \Psi^{\lambda_1} | \Psi^{\lambda_2} f | \big ]`.
        C11 (List[jnp.float64]): Sixth order covariance statistic :math:`\text{Cov}\big [ \Psi^{\lambda_1} | \Psi^{\lambda_3} f |, \Psi^{\lambda_1} | \Psi^{\lambda_2} f | \big ]`.
        N (List[jnp.ndarray]): Multiscale normalisation factors for given signal.
        J_min (int): Minimum dyadic wavelet scale to consider.
        J_max (int): Maximum dyadic wavelet scale to consider.

    Returns:
        Tuple[List[jnp.ndarray]]: Tuple of normalised scattering covariance statistics.
    """
    for j2 in range(J_min, J_max + 1):
        idx = j2 - J_min
        S1[idx] /= jnp.sqrt(N[idx])
        P00[idx] /= N[idx]

    for j1 in range(J_min, J_max):
        idx = j1 - J_min
        norm = jnp.einsum(
            "j,n->jn",
            1 / jnp.sqrt(N[idx]),
            1 / jnp.sqrt(N[idx]),
            optimize=True,
        )

        C01[idx] = jnp.einsum("ajn,jn->ajn", C01[idx], norm, optimize=True)
        C11[idx] = jnp.einsum("abjkn,jk->abjkn", C11[idx], norm, optimize=True)

    return S1, P00, C01, C11
