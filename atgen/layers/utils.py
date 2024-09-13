'''
this namespace to distinguish between different types of layers:
some layers requires modifying the following layers, and some other doesn't requires,
also when initializing identity element we need to distinguish where to stop and which layers to copy  
for identity element:
    - copy every thing from `Evolve` to `Activation`
    - if there is `Follow`, `Copy` or `Activation` layers between just copy both
    - if there is `Skip` layer, don't copy
for modification:
    - copy every thing from `Evolve` to next `Evolve`
    - if there is `Follow` layer edit it corresponding to the previous editing
    - if there is `Copy` or `Activation` layers between just copy both
    - if there is `Skip` layer, don't copy
    - if you reach the next `Evolve` layer modify its input corresponding to the previous editing
'''

from typing import List, Tuple
from torch import nn


class EvolveAction:
    IncreaseIn  = "IncreaseIn"            # i.e increase weights in linear or in_channel for cnn
    IncreaseOut = "IncreaseOut"           # i.e increase neurons in linear or out_channel for cnn
    DecreaseIn  = "DecreaseIn"            # i.e decrease weights in linear or in_channel for cnn
    DecreaseOut = "DecreaseOut"           # i.e decrease neurons in linear or out_channel for cnn


class LayerModifier:
    def __init__(self, follow: List[nn.Module]=[], copy: List[nn.Module]=[], skip: List[nn.Module]=[]):
        self.follow = follow
        self.copy = copy
        self.skip = skip

    def identity(self, layer: nn.Module) -> nn.Module:
        raise NotImplementedError(f"you need to implement the identity method for {layer.__class__.__name__}")

    def modify(self, layer: nn.Module, evolve_action: str) -> Tuple[nn.Module, bool]:
        raise NotImplementedError(f"you need to implement the modifier method for {layer.__class__.__name__}")
    
    def new(self, layer: nn.Module) -> nn.Module:
        raise NotImplementedError(f"you need to implement the new method for {layer.__class__.__name__}")

    

