[![image](https://github.com/astro-informatics/s2scat/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/astro-informatics/s2scat/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/astro-informatics/s2scat/graph/badge.svg?token=LTSRXQVHIA)](https://codecov.io/gh/astro-informatics/s2scat)
[![image](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![image](https://badge.fury.io/py/s2scat.svg)](https://badge.fury.io/py/s2scat)
[![image](http://img.shields.io/badge/arXiv-xxxx.xxxxx-orange.svg?style=flat)](https://arxiv.org/abs/xxxx.xxxxx)
[![All Contributors](https://img.shields.io/github/all-contributors/astro-informatics/s2scat?color=ee8449&style=flat-square)](#contributors)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/astro-informatics/s2scat/blob/main/notebooks/auto_generation.ipynb)

# Differentiable scattering covariances on the sphere

`S2SCAT` is a Python package for computing scattering covariances on the sphere ([Mousset et al. 2024](https://arxiv.org/abs/xxxx.xxxxx)) using JAX.  It exploits autodiff to provide differentiable transforms, which are also deployable on hardware accelerators (e.g. GPUs and TPUs), leveraging the differentiable and accelerated spherical harmonic and wavelet transforms implemented in [S2FFT](https://github.com/astro-informatics/s2fft) and [S2WAV](https://github.com/astro-informatics/s2wav), respectively. Scattering covariances are useful both for field-level generative modelling of complex non-Gaussian textures and for statistical compression of high dimensional field-level data, a key step of e.g. simulation based inference.

> [!IMPORTANT]
> It is worth highlighting that the input to `S2SCAT` are spherical harmonic coefficients, which can be generated with whichever software package you prefer, e.g. [`S2FFT`](https://github.com/astro-informatics/s2fft) or [`healpy`](https://healpy.readthedocs.io/en/latest/). Just ensure your harmonic coefficients are indexed using our convention; helper functions for this reindexing can be found in [`S2FFT`](https://github.com/astro-informatics/s2fft).

> [!TIP]
> At launch `S2SCAT` provides two core transform modes: on-the-fly, which performs underlying spherical harmonic and Wigner transforms through the [Price & McEwen](https://arxiv.org/abs/2311.14670) recursion; and precompute, which a priori computes and caches all Wigner elements required. The precompute approach will be faster but can only be run up to $L \sim 512$, whereas the on-the-fly approach will run up to $L \sim 2048$ and potentially beyond, depending on GPU hardware.

Ballpark compute times (when running on an 40GB A100 GPU) and compression levels are given in the table below. 

| Method | Resolution | Forward pass | Gradient pass | JIT compilation | Input params | Anisotropic  (compression) | Isotropic  (compression) |
|:----------------------------:|:--------------:|:------------:|:-------------:|:---------------:|:------------:|:--------------------------:|:------------------------:|
|           Precompute          |   L=512, N=3   |     ~90ms    |     ~190ms    |       ~20s      |   2,618,880  |     ~ 63,000  (97.594%)    |      ~504  (99.981%)     |
|          On-the-fly          |   L=2048, N=3  |     ~18s     |      ~40s     |       ~5m       |  41,932,800  |    ~ 123,750  (99.705%)    |     ~ 990  (99.998%)     |

Note that these times are not batched, so in practice may be substantially faster.

## Scattering covariances :dna:

<p align="center">
  <img width="300" height="300" src="./docs/assets/synthesis.gif">
</p>

We introduce scattering covariances on the sphere in [Mousset et al. (2024)](https://arxiv.org/abs/xxxx.xxxxx), which extend to spherical settings similar scattering transforms introduced for 1D signals by [Morel et al. (2023)](https://arxiv.org/abs/2204.10177) and for planar 2D signals by [Cheng et al. (2023)](https://arxiv.org/abs/2306.17210). 

Scattering covariances $S$ are computed by

$$S_1^{\lambda_1} = \langle |W^{\lambda_1} I| \rangle,$$

$$S_2^{\lambda_1} = \langle|W^{\lambda_1} I|^2 \rangle,$$

$$S_3^{\lambda_1, \lambda_2} = \text{Cov} \left[  W^{\lambda_1}I, W^{\lambda_1}|W^{\lambda_2} I| \right],$$

$$S_4^{\lambda_1, \lambda_2, \lambda_3} = \text{Cov} \left[W^{\lambda_1}|W^{\lambda_3}I|, W^{\lambda_1}|W^{\lambda_2}I|\right]$$

where $W^{\lambda} I$ denotes the wavelet transform of field $I$ at scale $j$ and direction $\gamma$, which we group into a single label $\lambda=(j,\gamma)$. 

This statistical representation characterises the power and sparsity at given scales, as well as covariant features between different wavelet scale and directions, which can effectively capture complex non-Gaussian structural information, e.g. filamentary structure.

Using the recently released JAX spherical harmonic code [`S2FFT`](https://github.com/astro-informatics/s2fft) ([Price & McEwen 2024](https://arxiv.org/abs/2311.14670)) and spherical wavelet transform code [`S2WAV`](https://github.com/astro-informatics/s2wav) ([Price et al. 2024](<https://arxiv.org/abs/2402.01282)) in the `S2SCAT` code we extends scattering covariances to the sphere, which are necessary for their application to generative modelling of wide-field cosmological fields ([Mousset et al. 2024](https://arxiv.org/abs/xxxx.xxxxx)).

## Usage :rocket:

To import and use `S2SCAT` is as simple follows:

``` python
import s2scat, jax
# For statistical compression
encoder = s2scat.build_encoder(L, N)          # Returns a callable compression model.
covariance_statistics = encoder(alm)          # Generate statistics (can be batched).

# For generative modelling
key = jax.random.PRNGKey(seed)
generator = s2scat.build_generator(alm, L, N) # Returns a callable generative model.
new_samples = generator(key, 10)              # Generate 10 new spherical textures. 
```

For further details on usage see the [documentation](https://astro-informatics.github.io/s2scat/) and associated [notebooks](https://astro-informatics.github.io/s2scat/notebooks/).

## Package Directory Structure :art:

``` bash
s2scat/  
├── representation.py   # - Scattering covariance transform.
├── compression.py      # - Statistical compression functions.
├── optimisation.py     # - Optimisation algorithm wrappers. 
├── generation.py       # - Latent encoder and Generative decoder.
│    
├── operators/          # Internal functionality:
│      ├─ spherical.py          # - Specific spherical operations, e.g. batched SHTs.
│      ├─ matrices.py           # - Wrappers to generate cached values. 
│
├── utility/            # Convenience functionality:
│      ├─ reorder.py            # - Reindexing and converting list and arrays.
│      ├─ statistics.py         # - Calculation of covariance statistics. 
│      ├─ normalisation.py      # - Normalisation functions for covariance statistics. 
│      ├─ plotting.py           # - Plotting functions for signals and statistics.
```

## Installation :computer:

The Python dependencies for the `S2SCAT` package are listed in the file
`requirements/requirements-core.txt` and will be automatically installed
into the active python environment by [pip](https://pypi.org) when running

``` bash
pip install s2scat
```
This will install all core functionality which includes full JAX support.

Alternatively, the `S2SCAT` package may be installed directly from GitHub by cloning this 
repository and then running 

``` bash
pip install .        
```

from the root directory of the repository. 

Unit tests can then be executed to ensure the installation was successful by first installing the test requirements and then running pytest

``` bash
pip install -r requirements/requirements-tests.txt
pytest tests/  
```

Documentation for the released version is available [here](https://astro-informatics.github.io/s2scat/).

## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://cosmomatt.github.io"><img src="https://avatars.githubusercontent.com/u/32554533?v=4?s=100" width="100px;" alt="Matt Price"/><br /><sub><b>Matt Price</b></sub></a><br /><a href="#ideas-CosmoMatt" title="Ideas, Planning, & Feedback">🤔</a> <a href="#code-CosmoMatt" title="Code">💻</a> <a href="#design-CosmoMatt" title="Design">🎨</a> <a href="#doc-CosmoMatt" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/mousset"><img src="https://avatars.githubusercontent.com/u/37935237?v=4?s=100" width="100px;" alt="mousset"/><br /><sub><b>mousset</b></sub></a><br /><a href="#code-mousset" title="Code">💻</a> <a href="#design-mousset" title="Design">🎨</a> <a href="#ideas-mousset" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://www.jasonmcewen.org"><img src="https://avatars.githubusercontent.com/u/3181701?v=4?s=100" width="100px;" alt="Jason McEwen "/><br /><sub><b>Jason McEwen </b></sub></a><br /><a href="#ideas-jasonmcewen" title="Ideas, Planning, & Feedback">🤔</a> <a href="#code-jasonmcewen" title="Code">💻</a> <a href="#doc-jasonmcewen" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Eralys"><img src="https://avatars.githubusercontent.com/u/47173968?v=4?s=100" width="100px;" alt="Eralys"/><br /><sub><b>Eralys</b></sub></a><br /><a href="#ideas-Eralys" title="Ideas, Planning, & Feedback">🤔</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->


## Attribution :books: 

Should this code be used in any way, we kindly request that the following article is
referenced. A BibTeX entry for this reference may look like:

```
    @article{mousset:s2scat, 
        author      = "Louise Mousset et al",
        title       = "TBD",
        journal     = "TBD, submitted",
        year        = "2024",
        eprint      = "TBD"        
    }
```

You might also like to consider citing our related papers on which this
code builds:

```
    @article{price:s2fft, 
        author      = "Matthew A. Price and Jason D. McEwen",         
        title        = "Differentiable and accelerated spherical harmonic and {W}igner transforms",
        journal      = "Journal of Computational Physics",
        volume       = "510",
        pages        = "113109",        
        year         = "2024",
        doi          = {10.1016/j.jcp.2024.113109},
        eprint       = "arXiv:2311.14670"        
    }
```
```
    @article{price:s2wav, 
        author      = "Matthew A. Price and Alicja Polanska and Jessica Whitney and Jason D. McEwen",
        title       = "Differentiable and accelerated directional wavelet transform on the sphere and ball",
        year        = "2024",
        eprint      = "arXiv:2402.01282"
    }
```

## License :memo:

We provide this code under an MIT open-source licence with the hope that
it will be of use to a wider community.

Copyright 2024 Louise Mousset, Matthew Price, Erwan Allys and Jason McEwen

`S2SCAT` is free software made available under the MIT License. For
details see the LICENSE file.
