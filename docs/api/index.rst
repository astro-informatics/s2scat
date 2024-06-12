:html_theme.sidebar_secondary.remove:

**************************
API
**************************
Automatically generated documentation for ``S2SCAT`` APIs. All functionality is accessible 
through a pip installation of the ``S2SCAT`` package. Below is an overview of the 
directory structure for the software.

Directory structure 
--------------------

.. code-block:: bash

   s2scat/  
   ├── representation.py   # - Scattering covariance transform.
   ├── compression.py      # - Statistical compression functions.
   ├── generation.py       # - Generative optimisation wrappers. 
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

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Namespaces
   
   core/index
   operators/index
   utility/index
