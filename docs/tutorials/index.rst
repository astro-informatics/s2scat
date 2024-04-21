:html_theme.sidebar_secondary.remove:

**************************
Notebooks
**************************
A series of tutorial notebooks which go through the absolute base level application of 
``S2SCAT`` apis. Post alpha release we will add examples for more involved applications, 
in the time being feel free to contact contributors for advice! At a high-level the 
``S2SCAT`` package is structured such that the 2 primary methods, emulation and compression, 
can easily be accessed.

Core usage |:rocket:|
-----------------
To import and use ``S2SCAT``  is as simple follows: 

.. code-block:: python

   import s2scat
   L = _   # Harmonic bandlimit 
   N = _   # Azimuthal bandlimit 
   flm = _ # Harmonic coefficients of the input signal 

   # Core GPU transforms 
   config = s2scat.configure(L, N)
   covariances = s2scat.scatter(flm, L, N, config=config)

   # C backend CPU transforms
   config = s2scat.configure(L, N, c_backend=True)
   covariances = s2scat.scatter_c(flm, L, N, config=config)

``S2SCAT`` also provides JAX support for existing C backend libraries which are memory efficient but CPU bound; at launch we support `SSHT <https://github.com/astro-informatics/ssht>`_, however this could be extended straightforwardly. This works by wrapping python bindings with custom JAX frontends.

For further details on usage see the `documentation <https://astro-informatics.github.io/s2scat/>`_ and associated `notebooks <https://astro-informatics.github.io/s2scat/notebooks/>`_.


.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Jupyter Notebooks

   synthesis/synthesis.nblink