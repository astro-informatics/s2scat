from jax import jit, grad
import jax.numpy as jnp
from functools import partial
from typing import List, Tuple
import time

import jaxopt, optax


def fit_jaxopt_scipy(
    params: jnp.ndarray,
    loss_func,
    method="L-BFGS-B",
    niter: int = 10,
    loss_history: list = None,
    apply_jit: bool = False,
    print_iters: int = 10,
) -> Tuple[jnp.ndarray, List]:
    """Minimises the declared loss function starting at params using jaxopt.

    Args:
        params (jnp.ndarray): Initial estimate (signal).
        loss_func (function): Loss function to minimise.
        method (str, optional): jaxopt optimization algorithm. Defaults to "L-BFGS-B".
        niter (int, optional): Maximum number of iterations. Defaults to 10.
        loss_history (list, optional): A list in which to store the loss history. Defaults to None.
        apply_jit (bool, optional): Whether to jit the training step. Defaults to False.
        print_iters (int, optional): How often to return the loss during training. Defaults to 10.

    Returns:
        Tuple[jnp.ndarray, List]: Optimised solution and loss history.
    """
    if loss_history is None:
        loss_history = []
        loss_history.append(loss_func(params))

    optimizer = jaxopt.ScipyMinimize(
        fun=loss_func, method=method, jit=apply_jit, maxiter=1
    )

    for i in range(niter):
        start = time.time()
        params, opt_state = optimizer.run(params)
        end = time.time()
        if i % print_iters == 0:
            loss_history.append(opt_state.fun_val)
            print(
                f"Iter {i}, Success: {opt_state.success}, Loss = {opt_state.fun_val}, Time = {end - start:.5f} s/iter"
            )

    return params, loss_history


def fit_optax(
    params: optax.Params,
    loss_func,
    niter: int = 10,
    loss_history: list = None,
    print_iters: int = 10,
    learning_rate: jnp.float32 = 1e-4,
    apply_jit: bool = False,
) -> Tuple[optax.Params, List]:
    """Minimises the declared loss function starting at params using optax (adam).

    Args:
        params (jnp.ndarray): Initial estimate (signal).
        loss_func (function): Loss function to minimise.
        method (str, optional): jaxopt optimization algorithm. Defaults to "L-BFGS-B".
        niter (int, optional): Maximum number of iterations. Defaults to 10.
        loss_history (list, optional): A list in which to store the loss history. Defaults to None.
        print_iters (int, optional): How often to return the loss during training. Defaults to 10.
        learning_rate (jnp.float32, optional): Adam learning rate for optax. Defaults to 1e-4.
        apply_jit (bool, optional): Whether to jit the training step. Defaults to False.

    Returns:
        Tuple[optax.Params, List]: Optimised solution and loss history.
    """

    grad_func = jit(grad(loss_func)) if apply_jit else grad(loss_func)

    if loss_history is None:
        loss_history = []

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    for i in range(niter):
        start = time.time()
        grads = jnp.conj(grad_func(params))
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        end = time.time()

        if i % print_iters == 0:
            loss_value = loss_func(params)
            loss_history.append(loss_value)
            print(f"Iter {i}, Loss: {loss_value:.10f}, Time = {end - start:.5f} s/iter")

    return params, loss_history


@jit
def l2_covariance_loss(predicts, targets) -> jnp.float64:
    """L2 loss wrapper for the scattering covariance.

    Args:
        predicts (List[jnp.ndarray]): Predicted scattering covariances.
        targets (List[jnp.ndarray]): Target scattering covariances.

    Returns:
        jnp.float64: L2 loss.
    """
    loss = 0
    for i in range(6):
        loss += l2_loss(predicts[i], targets[i])
    return loss


@jit
def l2_loss(predict, target) -> jnp.float64:
    """L2 loss for a single scattering covariance.

    Args:
        predict (jnp.ndarray): Predicted scattering covariance.
        target (jnp.ndarray): Target scattering covariance.

    Returns:
        jnp.float64: L2 loss.
    """
    return jnp.mean(jnp.abs(predict - target) ** 2)


@partial(jit, static_argnums=(2))
def get_P00prime(
    flm: jnp.ndarray,
    filter_linear: Tuple[jnp.ndarray],
    normalisation: List[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray]:
    """Computes P00prime which is the averaged power within each wavelet scale.

    Args:
        flm (jnp.ndarray): Spherical harmonic coefficients of signal.
        filter_linear (Tuple[jnp.ndarray]): Linearised wavelet filters.
        normalisation (List[jnp.ndarray], optional): _description_. Defaults to None.

    Returns:
        Tuple[jnp.ndarray]: Tuple of the power and averaged power over wavelet scales.
    """
    P00prime_ell = jnp.sum(
        jnp.abs(flm[None, :, :] * filter_linear[:, :, None]) ** 2, axis=2
    )
    P00prime = jnp.mean(P00prime_ell, axis=1)
    if normalisation is not None:
        P00prime /= normalisation
    return P00prime_ell, P00prime
