:html_theme.sidebar_secondary.remove:

**************************
Helper Functions
**************************

.. list-table:: Statistics Functions
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2scat.utility.statistics.compute_mean_variance`
     - Computes the mean and variance of spherical harmonic coefficients :math:`f_{\ell m}`.
   * - :func:`~s2scat.utility.statistics.normalize_map`
     - Normalises a spherical map to zero mean and unit variance.
   * - :func:`~s2scat.utility.statistics.compute_P00`
     - Stand alone function to compute the second order power statistics.
   * - :func:`~s2scat.utility.statistics.compute_C01_and_C11`
     - Stand alone function to compute the fourth and sixth order covariance statistics.
   * - :func:`~s2scat.utility.statistics.add_to_S1`
     - Computes and appends the mean field statistic :math:`\text{S1}_j = \langle |\Psi^\lambda f| \rangle` at scale :math:`j`.
   * - :func:`~s2scat.utility.statistics.add_to_P00`
     - Computes and appends the second order power statistic :math:`\text{P00}_j = \langle |\Psi^\lambda f|^2 \rangle` at scale :math:`j`.
   * - :func:`~s2scat.utility.statistics.add_to_C01`
     - Computes and appends the fourth order covariance statistic :math:`\text{C01}_j = \text{Cov}\big [ \Psi^{\lambda_1} f, \Psi^{\lambda_1} | \Psi^{\lambda_2} f | \big ]` at scale :math:`j`.
   * - :func:`~s2scat.utility.statistics.add_to_C11`
     - Computes and appends the sixth order covariance statistic :math:`\text{C11}_j = \text{Cov}\big [ \Psi^{\lambda_1} | \Psi^{\lambda_3} f |, \Psi^{\lambda_1} | \Psi^{\lambda_2} f | \big ]` at scale :math:`j`.
   * - :func:`~s2scat.utility.statistics.apply_norm`
     - Applies normalisation to a complete list of covariance statistics.

.. list-table:: Reindexing Functions 
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2scat.utility.reorder.nested_list_to_list_of_arrays`
     - Specific reindexing function which switches covariance wavelet scale list ordering.
   * - :func:`~s2scat.utility.reorder.list_to_array`
     - Converts list of covariance statistics to array of statistics for e.g. Loss functions.

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Helper Functions

   statistics
   reorder 
   