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

      some code here.

C backend usage |:bulb:|
-----------------
``S2SCAT`` also provides JAX support for existing C libraries, at launch this includes `SSHT <https://github.com/astro-informatics/ssht>`_. 
This works by wrapping python bindings with custom JAX frontends. Note that currently this C to JAX interoperability is currently 
limited to CPU, however for many applications this is desirable due to memory constraints.

For example, one may call these alternate backends for the spherical harmonic transform by:

.. code-block:: python

   some code here. 

This JAX frontend supports out of the box reverse mode automatic differentiation, 
and under the hood is simply linking to the C packages you are familiar with. In this 
way ``S2SCAT`` enhances existing packages with gradient functionality for modern signal processing 
applications!


.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Jupyter Notebooks
