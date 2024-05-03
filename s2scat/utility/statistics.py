from jax import jit
import jax.numpy as jnp
from functools import partial
from typing import List, Tuple


@jit
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


@partial(jit, static_argnums=(3, 4))
def compute_C01_and_C11(
    Nj1j2: List[jnp.ndarray],
    W: List[jnp.ndarray],
    Q: List[jnp.ndarray],
    J_min: int,
    J_max: int,
) -> Tuple[jnp.ndarray]:
    r"""Compute the fourth (C01) and sixth (C11) order covariances.

    Args:
        Nj1j2 (List[jnp.ndarray]): Multiscale second order wavelet coefficients.
        W (List[jnp.ndarray]): Multiscale first order wavelet coefficients.
        Q (List[jnp.ndarray]): Multiscale quadrautre weights of given sampling pattern.
        J_min (int): Minimum dyadic wavelet scale to consider.
        J_max (int): Maximum dyadic wavelet scale to consider.

    Returns:
        Tuple[jnp.ndarray]: Fourth and sixth order scattering covariance statistics.

    Notes:
        The fourth order statistic :math:`\text{C01} = \text{Cov}\big [ \Psi^{\lambda_1} f, \Psi^{\lambda_1} | \Psi^{\lambda_2} f | \big ]`
        can be seen as the covariance between power on scales :math:`\lambda_1=(j_1,n_1)`
        and :math:`\lambda_2=(j_2,n_2)`, and is hence a fourth order statistic which is
        sensitive to directional multiscale structure, e.g. filaments. C11 behaves similarly,
        but correlates between three scales and directions and is consequently sixth order.
    """
    C01 = []
    C11 = []
    for j1 in range(J_min, J_max):
        idx = j1 - J_min
        C01 = add_to_C01(C01, Nj1j2[idx], W[idx], Q[idx])
        C11 = add_to_C11(C11, Nj1j2[idx], Q[idx])
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


@jit
def add_to_C01(
    C01: List[jnp.float64], Nj1j2: jnp.ndarray, W: jnp.ndarray, Q: jnp.ndarray
) -> List[jnp.float64]:
    r"""Computes the fourth order covariance statistic :math:`\text{C01}_j = \text{Cov}\big [ \Psi^{\lambda_1} f, \Psi^{\lambda_1} | \Psi^{\lambda_2} f | \big ]` at scale :math:`j`.

    Args:
        C01 (List[jnp.float64]): List in which to append the fourth order covariance statistic.
        Nj1j2 (List[jnp.ndarray]): Second order wavelet coefficients at scale :math:`j`.
        W (jnp.ndarray): Spherical signal at a single scale :math:`j`.
        Q (List[jnp.ndarray]): Quadrautre weights of given sampling pattern at scale :math:`j`.

    Returns:
        List[jnp.float64]: List into which :math:`\text{C01}_j` has been appended.
    """
    val = jnp.einsum("ajntp,ntp->ajntp", jnp.conj(Nj1j2), W, optimize=True)
    val = jnp.einsum("ajntp,t->ajn", val, Q, optimize=True)
    C01.append(jnp.real(val))
    return C01


@jit
def add_to_C11(
    C11: List[jnp.float64], Nj1j2: jnp.ndarray, Q: jnp.ndarray
) -> List[jnp.float64]:
    r"""Computes the sixth order covariance statistic :math:`\text{C11}_j = \text{Cov}\big [ \Psi^{\lambda_1} | \Psi^{\lambda_3} f |, \Psi^{\lambda_1} | \Psi^{\lambda_2} f | \big ]` at scale :math:`j`.

    Args:
        C11 (List[jnp.float64]): List in which to append the sixth order covariance statistic.
        Nj1j2 (List[jnp.ndarray]): Second order wavelet coefficients at scale :math:`j`.
        Q (List[jnp.ndarray]): Quadrautre weights of given sampling pattern at scale :math:`j`.

    Returns:
        List[jnp.float64]: List into which :math:`\text{C11}_j` has been appended.
    """
    val = jnp.einsum("ajntp,bkntp->abjkntp", Nj1j2, jnp.conj(Nj1j2), optimize=True)
    val = jnp.einsum("abjkntp,t->abjkn", val, Q, optimize=True)
    C11.append(jnp.real(val))
    return C11


@jit
def compute_snr(target: jnp.ndarray, predict: jnp.ndarray) -> jnp.float64:
    r"""Computes the recovered signal to noise ratio (SNR) in dB.

    Args:
        target (jnp.ndarray): Ground truth signal.
        predict (jnp.ndarray): Estimated/recovered signal.
    
    Returns:
        jnp.float64: Signal to noise ratio (dB)
    """
    temp = jnp.sqrt(jnp.mean(jnp.abs(target)**2))
    temp /= jnp.sqrt(jnp.mean(jnp.abs(target-predict)**2))
    return 20 * jnp.log10(temp)

@jit
def compute_pearson_correlation(target: jnp.ndarray, predict: jnp.ndarray) -> jnp.float64:
    r"""Computes the recovered signal pearson correlation coefficient (structural similarity).

    Args:
        target (jnp.ndarray): Ground truth signal.
        predict (jnp.ndarray): Estimated/recovered signal.
    
    Returns:
        jnp.float64: Pearson correlation coefficient.
    """
    predict_mean = jnp.mean(predict)
    target_mean = jnp.mean(target)

    numerator = jnp.sum((predict-predict_mean)*(target-target_mean))
    denominator = jnp.sqrt(jnp.sum((predict-predict_mean)**2))
    denominator *= jnp.sqrt(jnp.sum((target-target_mean)**2))
    return numerator/denominator
