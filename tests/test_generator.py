import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest
import s2scat

L_to_test = [8]
N_to_test = [3]
recursive_transform = [False, True]
isotropic = [False, True]


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("recursive", recursive_transform)
@pytest.mark.parametrize("isotropic", isotropic)
def test_generator(L: int, N: int, recursive: bool, isotropic: bool):
    xlm = jnp.array(
        np.random.randn(L, L) + 1j * np.random.randn(L, L), dtype=jnp.complex64
    )
    generator = s2scat.build_generator(xlm, L, N, 0, True, recursive, isotropic)
    key = jax.random.PRNGKey(0)
    xlm_new = generator(key, count=10, niter=10)
    assert xlm_new.shape == (10, L, 2 * L - 1)


@pytest.mark.parametrize("L", L_to_test)
@pytest.mark.parametrize("N", N_to_test)
@pytest.mark.parametrize("recursive", recursive_transform)
@pytest.mark.parametrize("isotropic", isotropic)
def test_encoder(L: int, N: int, recursive: bool, isotropic: bool):
    xlm = jnp.array(
        np.random.randn(10, L, L) + 1j * np.random.randn(10, L, L),
        dtype=jnp.complex128,
    )
    encoder = s2scat.build_encoder(L, N, 0, True, recursive, isotropic)
    latents = encoder(xlm)
    assert len(latents) == 6
