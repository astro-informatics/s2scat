:html_theme.sidebar_secondary.remove:

**************************
Internal Functions
**************************

.. list-table:: Spherical Operations
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2scat.operators.spherical.make_flm_full`
     - Reflects real harmonic coefficients to a complete set of coefficients using hermitian symmetry.
   * - :func:`~s2scat.operators.spherical.make_flm_real`
     - Compresses harmonic coefficients of a real signal into positive coefficients only which leverages hermitian symmetry.
   * - :func:`~s2scat.operators.spherical.quadrature`
     - Generates spherical quadrature weights associated with McEwen-Wiaux sampling.
    
.. list-table:: Precompute helpers
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2scat.operators.matrices.generate_precompute_matrices`
     - Generates the full set of precompute matrices for the scattering transform with :math:`\mathcal{O}(NL^3)` memory overhead.
   * - :func:`~s2scat.operators.matrices.generate_recursive_matrices`
     - Generates a small set of recursive matrices for underlying Wigner-d recursion algorithms with a modest :math:`\mathcal{O}(NL^2)` memory overhead.

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Internal Functions

   spherical
   matrices 
   