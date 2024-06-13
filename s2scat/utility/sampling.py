import jax.numpy as jnp
import s2fft 

def to_healpix(xlm: jnp.ndarray, L: int) -> jnp.ndarray:
    xlm = jnp.squeeze(xlm)
    return s2fft.sampling.s2_samples.flm_2d_to_hp(xlm, L)

def from_healpix(xlm: jnp.ndarray, L: int) -> jnp.ndarray:
    xlm = jnp.squeeze(xlm)
    return s2fft.sampling.s2_samples.flm_hp_to_2d(xlm, L)
