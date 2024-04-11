Differentiable and accelerated spherical transforms
===================================================

``S2SCAT`` is a Python package for computing third generation scattering covariances on the 
sphere `(Mousset et al 2024) <https://arxiv.org/abs/2311.14670>`_ using 
JAX or PyTorch. It leverages autodiff to provide differentiable transforms, which are 
also deployable on hardware accelerators (e.g. GPUs and TPUs).

.. tip::
    At launch ``S2SCAT`` also provides PyTorch implementations of underlying 
    precompute transforms. In future releases this support will be extended to our 
    on-the-fly algorithms. ``S2SCAT`` also provides JAX frontend support for the highly optimised 
    but CPU bound SSHT C backends. These can be useful when GPU resources are not available or 
    memory constraints are tight.

Third Generation Scattering Covariances |:zap:|
---------------------------------------------------------

Details about the transform here with nice animations!


Contributors ✨
-----------------------------------

Thanks goes to these wonderful people (`emoji
key <https://allcontributors.org/docs/en/emoji-key>`_):

.. raw:: html 

    <embed>
        <table>
            <tbody>
            <tr>
                <td align="center" valign="top" width="14.28%"><a href="https://cosmomatt.github.io"><img src="https://avatars.githubusercontent.com/u/32554533?v=4?s=100" width="100px;" alt="Matt Price"/><br /><sub><b>Matt Price</b></sub></a><br /><a href="#ideas-CosmoMatt" title="Ideas, Planning, & Feedback">🤔</a> <a href="#code-CosmoMatt" title="Code">💻</a> <a href="#design-CosmoMatt" title="Design">🎨</a> <a href="#doc-CosmoMatt" title="Documentation">📖</a></td>
                <td align="center" valign="top" width="14.28%"><a href="https://github.com/mousset"><img src="https://avatars.githubusercontent.com/u/37935237?v=4?s=100" width="100px;" alt="mousset"/><br /><sub><b>mousset</b></sub></a><br /><a href="#code-mousset" title="Code">💻</a> <a href="#design-mousset" title="Design">🎨</a> <a href="#ideas-mousset" title="Ideas, Planning, & Feedback">🤔</a></td>
                <td align="center" valign="top" width="14.28%"><a href="http://www.jasonmcewen.org"><img src="https://avatars.githubusercontent.com/u/3181701?v=4?s=100" width="100px;" alt="Jason McEwen "/><br /><sub><b>Jason McEwen </b></sub></a><br /><a href="#ideas-jasonmcewen" title="Ideas, Planning, & Feedback">🤔</a></td>
            </tr>
            </tbody>
        </table>
    </embed>

We encourage contributions from any interested developers.

Attribution |:books:|
------------------

Should this code be used in any way, we kindly request that the following
article is referenced. A BibTeX entry for this reference may look like:

.. code-block:: 

    @article{mousset:s2scat, 
        author      = "Louise Mousset et al",
        title       = "TBD",
        journal     = "Astronomy & Astrophysics, submitted",
        year        = "2024",
        eprint      = "TBD"        
    }

You might also like to consider citing our related papers on which this code builds:

.. code-block:: 

    @article{price:s2fft, 
        author      = "Matthew A. Price and Jason D. McEwen",
        title       = "Differentiable and accelerated spherical harmonic and Wigner transforms",
        journal     = "Journal of Computational Physics, submitted",
        year        = "2023",
        eprint      = "arXiv:2311.14670"        
    }

.. code-block::
   
    @article{price:s2wav, 
        author      = {Matthew A. Price and Alicja Polanska and Jessica Whitney and Jason D. McEwen},
        title       = {"Differentiable and accelerated directional wavelet transform on the sphere and ball"},
        eprint      = {arXiv:2402.01282},
        year        = {2024}
    }

License |:memo:|
----------------

We provide this code under an MIT open-source licence with the hope that
it will be of use to a wider community.

Copyright 2024 Matthew Price, Louise Mousset, Erwan Allys and Jason McEwen

``S2SCAT`` is free software made available under the MIT License. For
details see the LICENSE file.

.. bibliography:: 
    :notcited:
    :list: bullet

.. * :ref:`modindex`

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: User Guide

   user_guide/install

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Interactive Tutorials
   
   tutorials/index

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: API

   api/index

