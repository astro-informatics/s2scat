from jax import jit
import jax.numpy as jnp
from functools import partial
from typing import List, Tuple


@partial(jit, static_argnums=(1))
def compute_mean_variance(flm: jnp.ndarray, L: int) -> Tuple[jnp.float64, jnp.float64]:
    """some docstrings"""
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
    """some docstrings"""
    P00 = []
    for j2 in range(J_min, J_max + 1):
        P00 = add_to_P00(P00, W[j2 - J_min], Q[j2 - J_min])

    if N is not None:
        for j2 in range(J_min, J_max + 1):
            P00[j2 - J_min] /= N[j2 - J_min]
    P00 = jnp.concatenate(P00) if S else P00
    return P00


@partial(jit, static_argnums=(3, 4))
def compute_C01_and_C11(
    Nj1j2: List[jnp.ndarray],
    W: List[jnp.ndarray],
    Q: List[jnp.ndarray],
    J_min: int,
    J_max: int,
) -> jnp.ndarray:
    """some docstrings"""
    C01 = []
    C11 = []
    for j1 in range(J_min, J_max):
        idx = j1 - J_min
        C01 = add_to_C01(C01, Nj1j2[idx], W[idx], Q[idx])
        C11 = add_to_C11(C11, Nj1j2[idx], Q[idx])
    return C01, C11


@partial(jit, static_argnums=(2))
def add_to_S1(S1: List[jnp.float64], Mlm: jnp.ndarray, L: int) -> List[jnp.float64]:
    """some docstrings"""
    val = Mlm[:, 0, L - 1] / (2 * jnp.sqrt(jnp.pi))
    S1.append(jnp.real(val))
    return S1


@jit
def add_to_P00(
    P00: List[jnp.float64], W: jnp.ndarray, Q: jnp.ndarray
) -> List[jnp.float64]:
    """some docstrings"""
    val = jnp.sum((jnp.abs(W) ** 2) * Q[None, :, None], axis=(-1, -2)) / (4 * jnp.pi)
    P00.append(jnp.real(val))
    return P00


@jit
def add_to_C01(
    C01: List[jnp.float64], Nj1j2: jnp.ndarray, W: jnp.ndarray, Q: jnp.ndarray
) -> List[jnp.float64]:
    """some docstrings"""
    val = jnp.einsum("ajntp,ntp,t->ajn", jnp.conj(Nj1j2), W, Q, optimize=True)
    C01.append(jnp.real(val))
    return C01


@jit
def add_to_C11(
    C11: List[jnp.float64], Nj1j2: jnp.ndarray, Q: jnp.ndarray
) -> List[jnp.float64]:
    """some docstrings"""
    val = jnp.einsum("ajntp,bkntp, t->abjkn", Nj1j2, jnp.conj(Nj1j2), Q, optimize=True)
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
) -> Tuple[List[jnp.ndarray], List[jnp.ndarray], List[jnp.ndarray], List[jnp.ndarray]]:
    """some docstrings"""
    for j2 in range(J_min, J_max + 1):
        S1[j2 - J_min] /= jnp.sqrt(N[j2 - J_min])
        P00[j2 - J_min] /= N[j2 - J_min]

    for j1 in range(J_min, J_max):
        norm = jnp.einsum(
            "j,n->jn",
            1 / jnp.sqrt(N[j1 - J_min]),
            1 / jnp.sqrt(N[j1 - J_min]),
            optimize=True,
        )

        C01[j1 - J_min] = jnp.einsum(
            "ajn,jn->ajn", C01[j1 - J_min], norm, optimize=True
        )
        C11[j1 - J_min] = jnp.einsum(
            "abjkn,jk->abjkn", C11[j1 - J_min], norm, optimize=True
        )

    return S1, P00, C01, C11
