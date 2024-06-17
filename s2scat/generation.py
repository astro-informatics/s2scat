import jax
import jax.numpy as jnp
from functools import partial
from typing import List, Callable
import s2scat


def build_encoder(
    L: int,
    N: int,
    J_min: int = 0,
    reality: bool = True,
    recursive: bool = False,
    isotropic: bool = False,
    delta_j: int = None,
    c_backend: bool = False,
) -> Callable:
    """Builds a scattering covariance encoding function.

    Args:
        L (int): Spherical harmonic bandlimit.
        N (int): Azimuthal bandlimit (directionality).
        J_min (int, optional): Minimum dyadic wavelet scale to consider. Defaults to 0.
        reality (bool, optional): Whether :math:`f \in \mathbb{R}`, if True exploits
            hermitian symmetry of harmonic coefficients. Defaults to True.
        recursive (bool, optional): Whether to perform a memory efficient recursive transform,
            or a faster but less memory efficient fully precompute transform. Defaults to False.
        isotropic (bool, optional): Whether to return isotropic coefficients, i.e. average
            over directionality. Defaults to False.
        delta_j (int, optional): Range of wavelet scales over which to compute covariances.
            If None, covariances between all scales will be considered. Defaults to None.
        c_backend (bool, optional): Whether to pick up and use the C backend functionality.
            Defaults to False.

    Returns:
        Callable: Latent encoder which takes arguements
            (xlm: jnp.ndarray) with index [batch, theta, phi].
    """

    # Compute and cache wavelet matrices
    config = s2scat.configure(L, N, J_min, reality, recursive, c_backend)

    # Partial function switch for encoder backend
    covariances = partial(
        s2scat.scatter_c if c_backend else s2scat.scatter,
        L=L,
        N=N,
        J_min=J_min,
        reality=reality,
        config=config,
        recursive=recursive,
        isotropic=isotropic,
        delta_j=delta_j,
    )

    # Define statistical encoder
    def encoder(xlm: jnp.ndarray) -> jnp.ndarray:
        if len(xlm.shape) > 2:
            return jax.vmap(covariances, in_axes=(0))(xlm)
        else:
            return covariances(xlm)

    return encoder


def build_generator(
    xlm: jnp.ndarray,
    L: int,
    N: int,
    J_min: int = 0,
    reality: bool = True,
    recursive: bool = False,
    isotropic: bool = False,
    delta_j: int = None,
    c_backend: bool = False,
) -> Callable:
    """Builds a scattering covariance generator function.

    Args:
        flm (jnp.ndarray): Spherical harmonic coefficients of target signal.
        L (int): Spherical harmonic bandlimit.
        N (int): Azimuthal bandlimit (directionality).
        J_min (int, optional): Minimum dyadic wavelet scale to consider. Defaults to 0.
        reality (bool, optional): Whether :math:`f \in \mathbb{R}`, if True exploits
            hermitian symmetry of harmonic coefficients. Defaults to True.
        recursive (bool, optional): Whether to perform a memory efficient recursive transform,
            or a faster but less memory efficient fully precompute transform. Defaults to False.
        isotropic (bool, optional): Whether to return isotropic coefficients, i.e. average
            over directionality. Defaults to False.
        delta_j (int, optional): Range of wavelet scales over which to compute covariances.
            If None, covariances between all scales will be considered. Defaults to None.
        c_backend (bool, optional): Whether to pick up and use the C backend functionality.
            Defaults to False.

    Returns:
        Callable: Latent decoder which takes arguements
            (key: jax.random.PRNGKey, count: int, niter: int = 400, learning_rate: float = 1e-3)
    """

    # Compute and cache wavelet matrices
    sigma = jnp.std(jnp.abs(xlm)[xlm != 0])
    config = s2scat.configure(L, N, J_min, reality, recursive, c_backend)

    # Compute normalisation
    norm = s2scat.compute_norm(xlm, L, N, J_min, reality, config, recursive)

    # Compute target statistics
    covariances = s2scat.scatter_c if c_backend else s2scat.scatter
    targets = covariances(
        xlm, L, N, J_min, reality, config, norm, recursive, isotropic, delta_j
    )

    def loss_function(xlm: jnp.ndarray) -> jnp.float64:
        """Standard L2 loss function to minimise during sampling"""
        predicts = covariances(
            xlm, L, N, J_min, reality, config, norm, recursive, isotropic, delta_j
        )
        return s2scat.optimisation.l2_covariance_loss(predicts, targets)

    # Define single sampler
    def sampler(
        xlm: jnp.ndarray, niter: int, learning_rate: float
    ) -> List[jnp.ndarray]:
        """Iterative sampling by adam gradient descent."""
        xlm = s2scat.optimisation.fit_optax(xlm, loss_function, niter, learning_rate)
        return s2scat.operators.spherical.make_flm_full(xlm, L) if reality else xlm

    # Define generative decoder
    def generator(
        key: jnp.ndarray, count: int, niter: int = 200, learning_rate: float = 1e-3
    ) -> List[jnp.ndarray]:
        """Generative iterative decoder."""
        xlm = _initial_arrays(key, sigma, L, count, reality)
        batched_sampler = jax.vmap(sampler, in_axes=(0, None, None))
        return batched_sampler(xlm, niter, learning_rate)

    return generator


def _initial_arrays(
    key: jnp.ndarray, sigma: float, L: int, count: int, reality: bool
) -> jnp.ndarray:
    """Private function which generates batched random vectors."""
    keys = jax.random.split(key, count)
    func = jax.vmap(_random_array, in_axes=(0, None, None))
    return func(keys, L, reality) * sigma


def _random_array(key: jnp.ndarray, L: int, reality: bool) -> jnp.ndarray:
    """Private function which generates individual random vectors."""
    keys = jax.random.split(key, 2)
    xlm = jax.random.normal(
        keys[0], (L, L), dtype=jnp.float64
    ) + 1j * jax.random.normal(keys[1], (L, L), dtype=jnp.float64)
    return xlm if reality else s2scat.operators.spherical.make_flm_full(xlm, L)
