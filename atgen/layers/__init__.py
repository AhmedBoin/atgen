import sys
import os

# Add the atgen directory to the Python path
sys.path.append(os.path.join(os.getcwd(), 'layers'))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .linear import LinearModifier
from .conv import Conv1dModifier, Conv2dModifier, Conv3dModifier
from .convtrans import ConvTranspose1dModifier, ConvTranspose2dModifier, ConvTranspose3dModifier
from .activations import ActiSwitch
from .normalization import (BatchNorm1dModifier, BatchNorm2dModifier, BatchNorm3dModifier, 
                            InstanceNorm1dModifier, InstanceNorm2dModifier, InstanceNorm3dModifier, 
                            LayerNormModifier, GroupNormModifier)
from .utils import EvolveAction, LayerModifier
