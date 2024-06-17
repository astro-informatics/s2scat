from . import representation
from . import optimisation
from . import compression
from . import generation

from . import utility
from . import operators

# Aliases
from .utility.config import run_config as configure
from .utility.normalisation import compute_norm

from .generation import build_encoder, build_generator

from .representation import scatter
from .representation import scatter_c
