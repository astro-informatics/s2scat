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

   import s2scat, jax

   # For statistical compression
   encoder = s2scat.build_encoder(L, N)        # Returns a callable compression model.
   covariance_statistics = encoder(alm)        # Generate statistics (can be batched).

   # For generative modelling
   key = jax.random.PRNGKey(seed)
   decoder = s2scat.build_decoder(alm, L, N)   # Returns a callable generative model.
   new_samples = decoder(key, 10)              # Generate 10 new spherical textures. 

``S2SCAT`` also provides JAX support for existing C backend libraries which are memory efficient but CPU bound; at launch we support `SSHT <https://github.com/astro-informatics/ssht>`_, however this could be extended straightforwardly. This works by wrapping python bindings with custom JAX frontends.

For further details on usage see the `documentation <https://astro-informatics.github.io/s2scat/>`_ and associated `notebooks <https://astro-informatics.github.io/s2scat/notebooks/>`_.


.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Jupyter Notebooks

   generation/automatic_generation.nblink
   generation/manual_generation.nblink
   compression/automatic_compression.nblink
   compression/manual_compression.nblink
