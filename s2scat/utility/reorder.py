from jax import jit
import jax.numpy as jnp
from functools import partial
from typing import List, Tuple


@partial(jit, static_argnums=(1, 2, 3))
def nested_list_to_list_of_arrays(
    Nj1j2: List[List[List[jnp.ndarray]]],
    N: int,
    J_min: int,
    J_max: int,
) -> List[jnp.ndarray]:
    """some docstrings"""
    Nj1j2_flat = []
    for j1 in range(J_min, J_max):
        Nj1j2_flat_for_j2 = []
        for j2 in range(j1 + 1, J_max + 1):
            Nj1j2_flat_for_j2.append(Nj1j2[j2 - J_min - 1][j1 - J_min])
        Nj1j2_flat.append(jnp.array(Nj1j2_flat_for_j2))
    return Nj1j2_flat


@jit
def list_to_array(
    S1: List[jnp.float64],
    P00: List[jnp.float64],
    C01: List[jnp.float64],
    C11: List[jnp.float64],
) -> Tuple[jnp.ndarray]:
    """some docstrings"""
    S1 = jnp.concatenate(S1)
    P00 = jnp.concatenate(P00)
    C01 = jnp.concatenate(C01, axis=None)
    C11 = jnp.concatenate(C11, axis=None)
    return S1, P00, C01, C11
