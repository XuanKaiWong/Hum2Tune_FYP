__version__ = "1.0.0"
__author__ = "FYP Student"
__description__ = "A system for recognizing songs from hummed melodies using frequency analysis"

# Import main components
from . import models
from . import tokenization
from . import utils

__all__ = [
    'models',
    'tokenization',
    'utils'
]