__version__ = '0.1.0'

__all__ = ['SCOAL', 'MSCOAL','EvoSCOAL','make_dataset']

from .algorithms._scoal import SCOAL
from .algorithms._mscoal import MSCOAL
from .algorithms._evoscoal import EvoSCOAL
from .data._make_dataset import make_dataset

