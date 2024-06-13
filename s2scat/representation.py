from jax import jit
import jax.numpy as jnp
from functools import partial
from typing import List

import s2wav
from s2scat.utility import statistics, reorder, normalisation
from s2scat.operators import spherical
from s2scat import compression

@partial(jit, static_argnums=(1, 2, 3, 4, 7, 8, 9))
def scatter(
    flm: jnp.ndarray,
    L: int,
    N: int,
    J_min: int = 0,
    reality: bool = True,
    config: List[jnp.ndarray] = None,
    norm: List[jnp.ndarray] = None,
    recursive: bool = False,
    isotropic: bool = False,
    delta_j: int = None,
) -> List[jnp.ndarray]:
    r"""Compute directional scattering covariances on the sphere.

    Args:
        flm (jnp.ndarray): Spherical harmonic coefficients.
        L (int): Spherical harmonic bandlimit.
        N (int): Azimuthal bandlimit (directionality).
        J_min (int, optional): Minimum dyadic wavelet scale to consider. Defaults to 0.
        reality (bool, optional): Whether :math:`f \in \mathbb{R}`, if True exploits
            hermitian symmetry of harmonic coefficients. Defaults to True.
        config (List[jnp.ndarray], optional): All necessary precomputed arrays. Defaults to None.
        norm (List[jnp.ndarray], optional): Covariance normalisation values.
            Defaults to None.
        recursive (bool, optional): Whether to perform a memory efficient recursive transform,
            or a faster but less memory efficient fully precompute transform. Defaults to False.
        isotropic (bool, optional): Whether to return isotropic coefficients, i.e. average
            over directionality. Defaults to False.
        delta_j (int, optional): Range of wavelet scales over which to compute covariances.
            If None, covariances between all scales will be considered. Defaults to None.

    Raises:
        ValueError: If one does not pass configuration arrays.

    Returns:
        Tuple[jnp.ndarray]: Directional scattering covariance statistics.

    Notes:
        The recursive transform, outlined in `Price & McEwen (2023) <https://arxiv.org/pdf/2311.14670>`_,
        requires :math:`\mathcal{O}(NL^2)` memory overhead and can scale to high bandlimits :math:`L`.
        Conversely, the fully precompute transform requires :math:`\mathcal{O}(NL^3)` memory overhead
        which can be large. However, the transform will be much faster. For applications at
        :math:`L \leq 512` the precompute approach is a better choice, beyond which we recommend the
        users switch to recursive transforms or the C backend functionality.

        If isotropic is true, the statistics will be contracted across :math:`n`. This will
        dramatically compress the covariance representation, but will be somewhat less
        sensitive to directional structure.
    """
    if config is None:
        raise ValueError("Must provide precomputed kernels for this transform!")
    filters, Q, precomps = config

    ### Configure maximum scale, impose reality, define quadrature
    J_max = s2wav.samples.j_max(L)
    Q = spherical.quadrature(L, J_min) if Q is None else Q
    flm = spherical.make_flm_full(flm, L) if reality else flm

    ### Compute: mean and Variance
    mean, var = statistics.compute_mean_variance(flm, L)

    ### Perform first wavelet transform W_j2 = f * Psi_j2
    W = spherical._first_flm_to_analysis(
        flm, L, N, J_min, reality, filters, precomps, recursive
    )

    ### Compute S1, P00, and Nj1j2
    Nj1j2, S1, P00 = [], [], []
    for j2 in range(J_min, J_max + 1):
        Lj2 = s2wav.samples.wav_j_bandlimit(L, j2, 2.0, True)

        ### Compute: Mlm = SHT(|W|)
        Mlm = spherical._forward_harmonic_vect(
            jnp.abs(W[j2 - J_min]), j2, Lj2, J_min, J_max, reality, precomps, recursive
        )

        ### Compute: S1 and P00 statistics
        S1 = statistics.add_to_S1(S1, Mlm, Lj2)
        P00 = statistics.add_to_P00(P00, W[j2 - J_min], Q[j2 - J_min])

        ### Compute: Nj1j2
        if j2 > J_min:
            val = spherical._flm_to_analysis_vect(
                Mlm,
                j2,
                Lj2,
                L,
                N,
                J_min,
                j2 - 1,
                reality,
                filters,
                precomps,
                recursive,
                delta_j,
            )
            Nj1j2.append(val)

    ### Reorder and flatten Njjprime, convert to JAX arrays for C01/C11
    Nj1j2_flat = reorder.nested_list_to_list_of_arrays(Nj1j2, J_min, J_max, delta_j)

    ### Compute: Higher order covariances C00/C11
    C01, C11 = statistics.compute_C01_and_C11(Nj1j2_flat, W, Q, J_min, J_max)

    ### Normalize the coefficients
    if norm is not None:
        S1, P00, C01, C11 = normalisation.apply_norm(
            S1, P00, C01, C11, norm, J_min, J_max
        )

    ### Compress covariances to isotropic coefficients
    if isotropic:
        C01, C11 = compression.C01_C11_to_isotropic(C01, C11, J_min, J_max)

    ### Return 1D jnp arrays for synthesis
    S1, P00, C01, C11 = reorder.list_to_array(S1, P00, C01, C11)

    return mean, var, S1, P00, C01, C11


def scatter_c(
    flm: jnp.ndarray,
    L: int,
    N: int,
    J_min: int = 0,
    reality: bool = False,
    config: List[jnp.ndarray] = None,
    norm: List[jnp.ndarray] = None,
    isotropic: bool = False,
    delta_j: int = None,
) -> List[jnp.ndarray]:
    r"""Compute directional scattering covariances on the sphere using a custom C backend.

    Args:
        flm (jnp.ndarray): Spherical harmonic coefficients.
        L (int): Spherical harmonic bandlimit.
        N (int): Azimuthal bandlimit (directionality).
        J_min (int, optional): Minimum dyadic wavelet scale to consider. Defaults to 0.
        reality (bool, optional): Whether :math:`f \in \mathbb{R}`, if True exploits
            hermitian symmetry of harmonic coefficients. Defaults to False.
        config (List[jnp.ndarray], optional): All necessary precomputed arrays. Defaults to None.
        norm (List[jnp.ndarray], optional): Covariance normalisation values.
            Defaults to None.
        isotropic (bool, optional): Whether to return isotropic coefficients, i.e. average
            over directionality. Defaults to False.
        delta_j (int, optional): Range of wavelet scales over which to compute covariances.
            If None, covariances between all scales will be considered. Defaults to None.

    Raises:
        ValueError: If one does not pass an array of wavelet filters.

    Returns:
        Tuple[jnp.ndarray]: Directional scattering covariance statistics.

    Notes:
        This variant of the directional scattering covariance transform leverages the
        JAX frontend for highly optimised C spherical harmonic libraries provided by
        `S2FFT <https://github.com/astro-informatics/s2fft/tree/main>`_. As such, it is
        currently limited to CPU compute and cannot be JIT compiled. However, this approach
        can still be very fast as the underlying spherical harmonic libraries are extremely
        optimised. Reverse mode gradient functionality is supported, peak memory overhead is
        :math:`\mathcal{O}(NL^2)`, and this variant can scale to very high :math:`L \geq 4096`.

        If isotropic is true, the statistics will be contracted across :math:`n`. This will
        dramatically compress the covariance representation, but will be somewhat less
        sensitive to directional structure.
    """
    if config is None:
        raise ValueError("Must provide precomputed kernels for this transform!")
    filters, Q, _ = config

    ### Configure maximum scale, impose reality, define quadrature
    J_max = s2wav.samples.j_max(L)
    Q = spherical.quadrature(L, J_min) if Q is None else Q
    flm = spherical.make_flm_full(flm, L) if reality else flm

    ### Compute: mean and Variance
    mean, var = statistics.compute_mean_variance(flm, L)

    ### Perform first wavelet transform W_j2 = f * Psi_j2
    W = spherical._first_flm_to_analysis(
        flm, L, N, J_min, reality, filters, use_c_backend=True
    )

    ### Compute S1, P00, and Nj1j2
    Nj1j2, S1, P00 = [], [], []
    for j2 in range(J_min, J_max + 1):
        Lj2 = s2wav.samples.wav_j_bandlimit(L, j2, 2.0, True)

        ### Compute: Mlm = SHT(|W|)
        Mlm = spherical._forward_harmonic_looped(jnp.abs(W[j2 - J_min]), Lj2, N)

        ### Compute: S1 and P00 statistics
        S1 = statistics.add_to_S1(S1, Mlm, Lj2)
        P00 = statistics.add_to_P00(P00, W[j2 - J_min], Q[j2 - J_min])

        ### Compute: Nj1j2
        if j2 > J_min:
            val = spherical._flm_to_analysis_looped(
                Mlm, j2, Lj2, L, N, J_min, j2 - 1, filters, delta_j
            )

            Nj1j2.append(val)

    ### Reorder and flatten Njjprime, convert to JAX arrays for C01/C11
    Nj1j2_flat = reorder.nested_list_to_list_of_arrays(Nj1j2, J_min, J_max, delta_j)

    ### Compute: Higher order covariances C00/C11
    C01, C11 = statistics.compute_C01_and_C11(Nj1j2_flat, W, Q, J_min, J_max)

    ### Normalize the coefficients
    if norm is not None:
        S1, P00, C01, C11 = normalisation.apply_norm(
            S1, P00, C01, C11, norm, J_min, J_max
        )

    ### Compress covariances to isotropic coefficients
    if isotropic:
        C01, C11 = compression.C01_C11_to_isotropic(C01, C11, J_min, J_max)

    ### Return 1D jnp arrays for synthesis
    S1, P00, C01, C11 = reorder.list_to_array(S1, P00, C01, C11)

    return mean, var, S1, P00, C01, C11
