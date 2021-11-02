__version__ = '0.1.0'

__all__ = ['SCOAL', 'MSCOAL','make_dataset']

from .algorithms._scoal import SCOAL
from .algorithms._mscoal import MSCOAL
from .data._make_dataset import make_dataset

