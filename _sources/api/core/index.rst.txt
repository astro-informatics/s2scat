:html_theme.sidebar_secondary.remove:

**************************
Core Functions
**************************

.. list-table:: Scattering Operations
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2scat.core.scatter.directional`
     - Compute directional scattering covariances on the sphere (Mousset et al 2024).
   * - :func:`~s2scat.core.scatter.directional_c`
     - Compute directional scattering covariances on the sphere using a custom C backend (Mousset et al 2024).

.. list-table:: Compression Operations
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2scat.core.compress.C01_C11_to_isotropic`
     - Convert scattering covariances to their isotropic counterpart.
    
.. list-table:: Synthesis Operations
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2scat.core.synthesis.fit_jaxopt_scipy`
     - Minimises the declared loss function starting at params using jaxopt.
   * - :func:`~s2scat.core.synthesis.fit_optax`
     - Minimises the declared loss function starting at params using optax (adam).
   * - :func:`~s2scat.core.synthesis.l2_covariance_loss`
     - L2 loss wrapper for the scattering covariance.
   * - :func:`~s2scat.core.synthesis.l2_loss`
     - L2 loss for a single scattering covariance.
   * - :func:`~s2scat.core.synthesis.get_P00prime`
     - Computes P00prime which is the averaged power within each wavelet scale.
    
.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Core Functions

   scatter
   compress 
   synthesis
   