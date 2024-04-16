import jax

jax.config.update("jax_enable_x64", True)
from jax.test_util import check_grads
import jax.numpy as jnp
import numpy as np
import pytest
import s2scat

import s2wav

L_to_test = [16]
N_to_test = [2, 3]
J_min_to_test = [0, 2]
reality_to_test = [False, True]
recursive_transform = [False, True]
delta_to_test = [None, 1]
isotropic = [False, True]


# This test uses the in built jax.check_grad function to validate gradients for a simple
# function within which we call the scattering transform. Thie ensures that the propagation
# of gradient information through the scattering transform is correct to at least ~1e-6.
@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("J_min", J_min_to_test)
@pytest.mark.parametrize("reality", reality_to_test)
@pytest.mark.parametrize("recursive", recursive_transform)
@pytest.mark.parametrize("delta_j", delta_to_test)
@pytest.mark.parametrize("isotropic", isotropic)
def test_gradients(
    L: int,
    N: int,
    J_min: int,
    reality: bool,
    recursive: bool,
    delta_j: int,
    isotropic: bool,
):
    J = s2wav.samples.j_max(L)

    # Exceptions
    if J_min >= J:
        pytest.skip("J_min larger than J which isn't a valid test case.")

    flm = jnp.array(np.random.randn(L, 2 * L - 1) + 1j * np.random.randn(L, 2 * L - 1))
    flm = s2scat.operators.spherical.make_flm_real(flm, L) if reality else flm

    filters = s2wav.filters.filters_directional_vectorised(L, N)[0]

    matrices = (
        s2scat.operators.matrices.generate_recursive_matrices(L, N, J_min, reality)
        if recursive
        else s2scat.operators.matrices.generate_precompute_matrices(
            L, N, J_min, reality
        )
    )

    def func(flm):
        coeffs = s2scat.core.scatter.directional(
            flm,
            L,
            N,
            J_min,
            reality,
            filters,
            None,
            None,
            matrices,
            recursive,
            isotropic,
            delta_j,
        )
        loss = 0
        for i in range(6):
            loss += jnp.mean(jnp.abs(coeffs[i]))
        return loss

    rtol = 5e-3 if isotropic else 1e-3
    check_grads(func, (flm,), order=1, modes=("rev"), rtol=rtol)
