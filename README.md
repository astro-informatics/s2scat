[![image](https://github.com/astro-informatics/s2scat/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/astro-informatics/s2scat/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/astro-informatics/s2scat/graph/badge.svg?token=LTSRXQVHIA)](https://codecov.io/gh/astro-informatics/s2scat)
[![image](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![image](https://badge.fury.io/py/s2scat.svg)](https://badge.fury.io/py/s2scat)
[![image](http://img.shields.io/badge/arXiv-xxxx.xxxxx-orange.svg?style=flat)](https://arxiv.org/abs/xxxx.xxxxx)
[![All Contributors](https://img.shields.io/github/all-contributors/astro-informatics/s2fft?color=ee8449&style=flat-square)](#contributors)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](add_link_here)

# S2SCAT: Scattering covariance transform on the sphere

<img align="center" src="./docs/assets/synthesis_zoom.gif">

`S2SCAT` is a Python package for computing third generation scattering covariances on the sphere [(Mousset et al 2024)](https://arxiv.org/abs/xxxx.xxxxx) using JAX. It leverages autodiff to provide differentiable transforms, which are also deployable on hardware accelerators (e.g. GPUs and TPUs). Scattering covariances are useful both for field-level emulation of complex non-Gaussian textures and for statistical compression of high dimensional field-level data, a key step of e.g. simulation based inference [(Cranmer et al 2020)](https://www.pnas.org/doi/abs/10.1073/pnas.1912789117).

> [!TIP]
> At launch `S2SCAT` provides JAX frontend support for the highly optimised but CPU bound SSHT C backends. These can be useful when GPU resources are not available or memory constraints are tight.


## Third Generation Scattering Statistics :dna:

<img align="right" width="300" height="300" src="./docs/assets/synthesis.gif">

Scattering covariances, or scattering spectra, were previously introduced for 1D signals by [Morel et al (2023)](https://arxiv.org/abs/2204.10177) and for planar 2D signals by [Cheng et al (2023)](https://arxiv.org/abs/2306.17210). The scattering transform is defined by repeated application of directional wavelet transforms followed by a machine learning inspired non-linearity, typically the modulus operator. The wavelet transform $W^{\lambda}$ within each layer has an associated scale $j$ and direction $n$, which we group into a single label $\lambda$. Scattering covariances $S$ are computed from the coefficients of a two-layer scattering transform and are defined as

$$S_1^{\lambda_1} = \langle |W^{\lambda_1} I| \rangle \quad S_2^{\lambda_1} = \langle|W^{\lambda_1} I|^2 \rangle$$

$$S_3^{\lambda_1, \lambda_2} = \text{Cov} \left[  W^{\lambda_1}I, W^{\lambda_1}|W^{\lambda_2} I| \right]$$

$$S_4^{\lambda_1, \lambda_2, \lambda_3} = \text{Cov} \left[W^{\lambda_1}|W^{\lambda_3}I|, W^{\lambda_1}|W^{\lambda_2}I|\right].$$

Given that the highest order coefficients are computed from products between $\lambda_1, \lambda_2$ and $\lambda_3$ they encode $6^{\text{th}}$-order statistical information. This statistical representation characterises the power and sparsity at given scales, as well as covariant features between different wavelet scale and directions; which can adequetly capture complex non-Gaussian structural information, e.g. filamentary structure. Using recently release JAX spherical harmonic [(Price & McEwen 2023)](https://arxiv.org/abs/2311.14670) and wavelet transforms [(Price et al 2024)](https://arxiv.org/abs/2402.01282) this work extends scattering covariances to the sphere, which is necessary for their application to e.g. wide-field cosmological surveys [(Mousset et al 2024)](https://arxiv.org/abs/xxxx.xxxxx).

## Package Directory Structure :art:

``` bash
s2scat/  
├── core/               # Top-level functionality:
│      ├─ scatter.py            # - Scattering covariance transform.
│      ├─ compress.py           # - Statistical compression functions.
│      ├─ synthesis.py          # - Synthesis optimisation functions. 
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

## Usage :rocket:

To import and use `S2SCAT` is as simple follows:

``` python
import s2scat, s2wav
L = _   # Harmonic bandlimit 
N = _   # Azimuthal bandlimit 
flm = _ # Harmonic coefficients of the input signal 

# Core GPU transforms 
config = s2scat.configure(L, N)
covariances = s2scat.scatter(flm, L, N, config=config)

# C backend CPU transforms
config = s2scat.configure(L, N, c_backend=True)
covariances = s2scat.scatter_c(flm, L, N, config=config)
```
`S2SCAT` also provides JAX support for existing C backend libraries which are memory efficient but CPU bound; at launch we support [`SSHT`](https://github.com/astro-informatics/ssht), however this could be extended straightforwardly. This works by wrapping python bindings with custom JAX frontends.

For further details on usage see the [documentation](https://astro-informatics.github.io/s2scat/) and associated [notebooks](add_link_here).

## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://cosmomatt.github.io"><img src="https://avatars.githubusercontent.com/u/32554533?v=4?s=100" width="100px;" alt="Matt Price"/><br /><sub><b>Matt Price</b></sub></a><br /><a href="#ideas-CosmoMatt" title="Ideas, Planning, & Feedback">🤔</a> <a href="#code-CosmoMatt" title="Code">💻</a> <a href="#design-CosmoMatt" title="Design">🎨</a> <a href="#doc-CosmoMatt" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/mousset"><img src="https://avatars.githubusercontent.com/u/37935237?v=4?s=100" width="100px;" alt="mousset"/><br /><sub><b>mousset</b></sub></a><br /><a href="#code-mousset" title="Code">💻</a> <a href="#design-mousset" title="Design">🎨</a> <a href="#ideas-mousset" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://www.jasonmcewen.org"><img src="https://avatars.githubusercontent.com/u/3181701?v=4?s=100" width="100px;" alt="Jason McEwen "/><br /><sub><b>Jason McEwen </b></sub></a><br /><a href="#ideas-jasonmcewen" title="Ideas, Planning, & Feedback">🤔</a></td>
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
        journal     = "Astronomy & Astrophysics, submitted",
        year        = "2024",
        eprint      = "TBD"        
    }
```

You might also like to consider citing our related papers on which this
code builds:

```
    @article{price:s2fft, 
        author      = "Matthew A. Price and Jason D. McEwen",
        title       = "Differentiable and accelerated spherical harmonic and Wigner transforms",
        journal     = "Journal of Computational Physics, submitted",
        year        = "2023",
        eprint      = "arXiv:2311.14670"        
    }
```
```
    @article{price:s2wav, 
        author      = {Matthew A. Price and Alicja Polanska and Jessica Whitney and Jason D. McEwen},
        title       = {"Differentiable and accelerated directional wavelet transform on the sphere and ball"},
        eprint      = {arXiv:2402.01282},
        year        = {2024}
    }
```

## License :memo:

We provide this code under an MIT open-source licence with the hope that
it will be of use to a wider community.

Copyright 2024 Louise Mousset, Matthew Price, Erwan Allys and Jason McEwen

`S2SCAT` is free software made available under the MIT License. For
details see the LICENSE file.
