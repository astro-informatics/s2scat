from jax import jit
import jax.numpy as jnp
from functools import partial
from typing import List

import s2wav
from s2scat.utility import sph_op, statistics, reorder


@partial(jit, static_argnums=(1, 2, 3, 4))
def directional(
    flm: jnp.ndarray,
    L: int,
    N: int,
    J_min: int = 0,
    reality: bool = False,
    filters: jnp.ndarray = None,
    normalisation: List[jnp.ndarray] = None,
    quads: List[jnp.ndarray] = None,
    precomps: List[List[jnp.ndarray]] = None,
) -> List[jnp.ndarray]:
    """some docstrings"""

    if precomps == None:
        raise ValueError("Must provide precomputed kernels for this transform!")

    # Configure maximum scale, impose reality, define quadrature
    J_max = s2wav.samples.j_max(L)
    quads = sph_op.quadrature(L, J_min, J_max) if quads is None else quads
    flm = sph_op.make_flm_full(flm, L) if reality else flm

    ### Compute: mean and Variance
    mean, var = statistics.compute_mean_variance(flm, L)

    ### Perform first wavelet transform W_j2 = f * Psi_j2
    W = sph_op._first_flm_to_analysis(flm, L, N, J_min, reality, filters, precomps)

    # Compute S1, P00, and Nj1j2
    Nj1j2 = []
    S1 = []
    P00 = []
    for j2 in range(J_min, J_max + 1):
        Lj2 = s2wav.samples.wav_j_bandlimit(L, j2, 2.0, True)

        # Compute: Mlm = SHT(|W|)
        Mlm = sph_op._forward_harmonic_vect(
            jnp.abs(W[j2 - J_min]), j2, Lj2, J_min, reality, precomps
        )

        ### Compute: S1 and P00 statistics
        S1 = statistics.add_to_S1(S1, Mlm, Lj2)
        P00 = statistics.add_to_P00(P00, W[j2 - J_min], quads[j2 - J_min])

        ### Compute: Nj1j2
        if j2 > J_min:
            val = sph_op._flm_to_analysis_vect(
                Mlm, j2, Lj2, L, N, J_min, j2 - 1, reality, filters, precomps
            )
            Nj1j2.append(val)

    ### Reorder and flatten Njjprime, convert to JAX arrays for C01/C11
    Nj1j2_flat = reorder.nested_list_to_list_of_arrays(Nj1j2, N, J_min, J_max)

    ### Compute: Higher order covariances C00/C11
    C01, C11 = statistics.compute_C01_and_C11(Nj1j2_flat, W, quads, J_min, J_max)

    ### Normalize the coefficients
    if normalisation is not None:
        S1, P00, C01, C11 = statistics.apply_norm(S1, P00, C01, C11, N, J_min, J_max)

    ### Return 1D jnp arrays for synthesis
    S1, P00, C01, C11 = reorder.list_to_array(S1, P00, C01, C11)

    return mean, var, S1, P00, C01, C11
