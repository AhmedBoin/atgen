import sys
import os

# Add the atgen directory to the Python path
sys.path.append(os.path.join(os.getcwd(), 'atgen'))


from .config import ATGENConfig
from .dna import DNA
from .species import Individual, Species
from .layers import *