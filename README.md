[![image](https://github.com/astro-informatics/s2scat/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/astro-informatics/s2scat/actions/workflows/tests.yml)
[![image](https://codecov.io/gh/astro-informatics/s2scat/branch/main/graph/badge.svg?token=7QYAFAAWLE)](https://codecov.io/gh/astro-informatics/s2scat)
[![image](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![image](https://badge.fury.io/py/s2scat.svg)](https://badge.fury.io/py/s2scat)
[![image](http://img.shields.io/badge/arXiv-xxxx.xxxxx-orange.svg?style=flat)](https://arxiv.org/abs/xxxx.xxxxx)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](add_link_here)
[![image](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Scattering covariance transform on the sphere

`S2SCAT` is a Python package for computing third generation scattering covariances on the 
sphere [(Mousset et al 2024)](https://arxiv.org/abs/2311.14670) using 
JAX or PyTorch. It leverages autodiff to provide differentiable transforms, which are 
also deployable on hardware accelerators (e.g. GPUs and TPUs).

> [!TIP]
At launch `S2SCAT` also provides PyTorch implementations of underlying 
precompute transforms. In future releases this support will be extended to our 
on-the-fly algorithms. `S2SCAT` also provides JAX frontend support for the highly optimised 
but CPU bound SSHT C backends. These can be useful when GPU resources are not available or 
memory constraints are tight.


## Third Generation Scattering Covariances :zap:

Details about the transform here with nice animations.


## Installation :computer:

The Python dependencies for the `S2SCAT` package are listed in the file
`requirements/requirements-core.txt` and will be automatically installed
into the active python environment by [pip](https://pypi.org) when running

``` bash
pip install s2scat
```
This will install all core functionality which includes JAX support (including PyTorch support).

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

Documentation for the released version is available [here](https://astro-informatics.github.io/s2scat/).  To build the documentation locally run

``` bash
pip install -r requirements/requirements-docs.txt
cd docs 
make html
open _build/html/index.html
```

## Usage :rocket:

To import and use `S2SCAT` is as simple follows:

``` python
Code example here. 
```

For further details on usage see the [documentation](https://astro-informatics.github.io/s2scat/) 
and associated [notebooks](add_link_here).

> [!NOTE]  
> We also provide PyTorch support for the precompute version of our transforms. These 
> are called through forward/inverse_torch(). Full PyTorch support will be provided in 
> future releases.

## C/C++ JAX Frontends for SSHT/HEALPix :bulb:

`S2SCAT` also provides JAX support for existing C backend libraries which are memory efficient 
but CPU bound; at launch we support [`SSHT`](https://github.com/astro-informatics/ssht), 
however this could be extended straightforwardly. This works by wrapping python bindings 
with custom JAX frontends.

For example, one may call these alternate backends for the spherical harmonic transform by:

``` python
Code example here. 
```

All of these JAX frontends supports out of the box reverse mode automatic differentiation, 
and under the hood is simply linking to the C/C++ packages you are familiar with. In this 
way `S2SCAT` supports existing backend transforms with gradient functionality for modern 
scientific computing or machine learning applications!

For further details on usage see the associated [notebooks](add_link_here).

## Contributors ✨

Thanks goes to these wonderful people ([emoji
key](https://allcontributors.org/docs/en/emoji-key)):
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://cosmomatt.github.io"><img src="https://avatars.githubusercontent.com/u/32554533?v=4?s=100" width="100px;" alt="Matt Price"/><br /><sub><b>Matt Price</b></sub></a><br /><a href="https://github.com/astro-informatics/s2fft/commits?author=CosmoMatt" title="Code">💻</a> <a href="https://github.com/astro-informatics/s2fft/pulls?q=is%3Apr+reviewed-by%3ACosmoMatt" title="Reviewed Pull Requests">👀</a> <a href="#ideas-CosmoMatt" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://www.jasonmcewen.org"><img src="https://avatars.githubusercontent.com/u/3181701?v=4?s=100" width="100px;" alt="Jason McEwen "/><br /><sub><b>Jason McEwen </b></sub></a><br /><a href="https://github.com/astro-informatics/s2fft/commits?author=jasonmcewen" title="Code">💻</a> <a href="https://github.com/astro-informatics/s2fft/pulls?q=is%3Apr+reviewed-by%3Ajasonmcewen" title="Reviewed Pull Requests">👀</a> <a href="#ideas-jasonmcewen" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://matt-graham.github.io"><img src="https://avatars.githubusercontent.com/u/6746980?v=4?s=100" width="100px;" alt="Matt Graham"/><br /><sub><b>Matt Graham</b></sub></a><br /><a href="https://github.com/astro-informatics/s2fft/commits?author=matt-graham" title="Code">💻</a> <a href="https://github.com/astro-informatics/s2fft/pulls?q=is%3Apr+reviewed-by%3Amatt-graham" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://sfmig.github.io/"><img src="https://avatars.githubusercontent.com/u/33267254?v=4?s=100" width="100px;" alt="sfmig"/><br /><sub><b>sfmig</b></sub></a><br /><a href="https://github.com/astro-informatics/s2fft/commits?author=sfmig" title="Code">💻</a> <a href="https://github.com/astro-informatics/s2fft/pulls?q=is%3Apr+reviewed-by%3Asfmig" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Devaraj-G"><img src="https://avatars.githubusercontent.com/u/36169767?v=4?s=100" width="100px;" alt="Devaraj Gopinathan"/><br /><sub><b>Devaraj Gopinathan</b></sub></a><br /><a href="https://github.com/astro-informatics/s2fft/commits?author=Devaraj-G" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://flanusse.net"><img src="https://avatars.githubusercontent.com/u/861591?v=4?s=100" width="100px;" alt="Francois Lanusse"/><br /><sub><b>Francois Lanusse</b></sub></a><br /><a href="https://github.com/astro-informatics/s2fft/commits?author=EiffL" title="Code">💻</a> <a href="https://github.com/astro-informatics/s2fft/issues?q=author%3AEiffL" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/eltociear"><img src="https://avatars.githubusercontent.com/u/22633385?v=4?s=100" width="100px;" alt="Ikko Eltociear Ashimine"/><br /><sub><b>Ikko Eltociear Ashimine</b></sub></a><br /><a href="https://github.com/astro-informatics/s2fft/commits?author=eltociear" title="Documentation">📖</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/kmulderdas"><img src="https://avatars.githubusercontent.com/u/33317219?v=4?s=100" width="100px;" alt="Kevin Mulder"/><br /><sub><b>Kevin Mulder</b></sub></a><br /><a href="https://github.com/astro-informatics/s2fft/issues?q=author%3Akmulderdas" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/PhilippMisofCH"><img src="https://avatars.githubusercontent.com/u/142883157?v=4?s=100" width="100px;" alt="Philipp Misof"/><br /><sub><b>Philipp Misof</b></sub></a><br /><a href="https://github.com/astro-informatics/s2fft/issues?q=author%3APhilippMisofCH" title="Bug reports">🐛</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
We encourage contributions from any interested developers. A simple
first addition could be adding support for more spherical sampling
patterns!

## Attribution :books: 

Should this code be used in any way, we kindly request that the following article is
referenced. A BibTeX entry for this reference may look like:

```
    @article{mousset:s2scat, 
        author      = "Louise Mousset, Matthew A. Price, Erwan Allys and Jason D. McEwen",
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

`S2FFT` is free software made available under the MIT License. For
details see the LICENSE file.
