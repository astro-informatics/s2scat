import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest
import s2scat

import s2wav

L_to_test = [8]
N_to_test = [3]
J_min_to_test = [0, 1]
delta_to_test = [None, 1]
isotropic = [False, True]
normalise = [False, True]

# Both GPU directional transforms are built from the same core recursion relations,
# hence it is reasonable to expect that the scattering representation should match
# to a decent level of precision (say around 1e-5).
@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("J_min", J_min_to_test)
@pytest.mark.parametrize("delta_j", delta_to_test)
@pytest.mark.parametrize("isotropic", isotropic)
@pytest.mark.parametrize("normalise", normalise)
def test_forward_pass(
    L: int, N: int, J_min: int, delta_j: int, isotropic: bool, normalise: bool
):
    J = s2wav.samples.j_max(L)
    reality = False
    # Exceptions
    if J_min >= J:
        pytest.skip("J_min larger than J which isn't a valid test case.")

    flm = jnp.array(np.random.randn(L, 2 * L - 1) + 1j * np.random.randn(L, 2 * L - 1))
    flm = s2scat.operators.spherical.make_flm_real(flm, L) if reality else flm

    config_precompute = s2scat.utility.config.run_config(
        L, N, J_min, reality, False, False
    )
    config_recursive = s2scat.utility.config.run_config(
        L, N, J_min, reality, True, False
    )

    norm = (
        s2scat.utility.normalisation.compute_norm(
            flm, L, N, J_min, reality, config_precompute, False
        )
        if normalise
        else None
    )

    coeffs_recursive = s2scat.scatter(
        flm, L, N, J_min, reality, config_recursive, norm, True, isotropic, delta_j
    )
    coeffs_precompute = s2scat.scatter(
        flm, L, N, J_min, reality, config_precompute, norm, False, isotropic, delta_j
    )

    for i in range(6):
        np.testing.assert_allclose(coeffs_recursive[i], coeffs_precompute[i])


# Directional_c function is built on slightly different backend recursions. If these
# differ by ~1e-6 then after multiple passes, e.g. layers in the scattering transform,
# the differences compound and can quite reasonable reach say ~1e-3.
# Additionally, the directional_c functions internally do not support lower bounding of
# the harmonic half-line during multiscale transforms. Therefore, directly the computed
# statistics are NOT theoretically the same, which is found in practice. Therefore,
# this test just checks that they are in the same ball-park.
@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("J_min", J_min_to_test)
@pytest.mark.parametrize("delta_j", delta_to_test)
@pytest.mark.parametrize("isotropic", isotropic)
def test_forward_pass_c_backend(
    L: int, N: int, J_min: int, delta_j: int, isotropic: bool
):
    J = s2wav.samples.j_max(L)
    reality = False
    # Exceptions
    if J_min >= J:
        pytest.skip("J_min larger than J which isn't a valid test case.")

    flm = jnp.array(np.random.randn(L, 2 * L - 1) + 1j * np.random.randn(L, 2 * L - 1))
    flm = s2scat.operators.spherical.make_flm_real(flm, L) if reality else flm

    config_c = s2scat.utility.config.run_config(L, N, J_min, reality, True, True)
    config_precompute = s2scat.utility.config.run_config(
        L, N, J_min, reality, False, False
    )
    coeffs_precompute = s2scat.scatter(
        flm, L, N, J_min, reality, config_precompute, None, False, isotropic, delta_j
    )

    coeffs_c = s2scat.scatter_c(
        flm, L, N, J_min, reality, config_c, None, isotropic, delta_j
    )

    for i in range(6):
        np.testing.assert_allclose(coeffs_precompute[i], coeffs_c[i], atol=1e-3)
