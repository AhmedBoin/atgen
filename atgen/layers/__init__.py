import sys
import os

# Add the atgen directory to the Python path
sys.path.append(os.path.join(os.getcwd(), 'layers'))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .linear import Linear, Flatten, LazyLinear
from .conv import Conv2D, MaxPool2D, LazyConv2D
from .activations import ActiSwitch, Pass
