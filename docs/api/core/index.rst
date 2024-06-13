:html_theme.sidebar_secondary.remove:

**************************
Core Functions
**************************

.. list-table:: Generative models
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2scat.generation.build_encoder`
     - Builds a scattering covariance encoding function (latent encoder).
   * - :func:`~s2scat.generation.build_decoder`
     - Builds a scattering covariance decoder function. (generative decoder).

.. list-table:: Scattering transforms
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2scat.representation.scatter`
     - Compute directional scattering covariances on the sphere (Mousset et al 2024).
   * - :func:`~s2scat.representation.scatter_c`
     - Compute directional scattering covariances on the sphere using a custom C backend (Mousset et al 2024).

.. list-table:: Compression functions
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2scat.compression.C01_C11_to_isotropic`
     - Convert scattering covariances to their isotropic counterpart.
    
.. list-table:: Optimisation functions
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2scat.optimisation.fit_optax`
     - Minimises the declared loss function starting at params using optax (adam).
   * - :func:`~s2scat.optimisation.l2_covariance_loss`
     - L2 loss wrapper for the scattering covariance.
   * - :func:`~s2scat.optimisation.l2_loss`
     - L2 loss for a single scattering covariance.
   * - :func:`~s2scat.optimisation.get_P00prime`
     - Computes P00prime which is the averaged power within each wavelet scale.
    
.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Core Functions

   generation
   representation
   compression
   optimisation

   