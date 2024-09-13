import copy
import math
from typing import List, Tuple
import torch
from torch import nn

from .utils import EvolveAction, LayerModifier

class LinearModifier(LayerModifier):
    def __init__(self, follow: List[nn.Module]=[], copy: List[nn.Module]=[], skip: List[nn.Module]=[]):
        super().__init__(follow, copy, skip)
        
    def new(self, layer: nn.Module) -> nn.Linear:
        if isinstance(layer, nn.Linear):
            layer = copy.deepcopy(layer)
            return nn.Linear(
                in_features=layer.in_features,
                out_features=layer.out_features,
                bias=False if layer.bias is None else True,
                dtype=layer.weight.data.dtype,
                device=layer.weight.data.device
            )

    def identity(self, layer: nn.Module) -> nn.Linear:
        if isinstance(layer, nn.Linear):
            layer = copy.deepcopy(layer)

            size = layer.weight.data.shape[1]
            bias = False if layer.bias is None else True
            dtype = layer.weight.data.dtype
            device = layer.weight.data.device

            layer = nn.Linear(in_features=size, out_features=size, bias=bias, dtype=dtype, device=device)

            layer.weight.data = torch.eye(size, size, dtype=dtype, device=device)
            if bias:
                layer.bias.data = torch.zeros_like(layer.bias.data, dtype=dtype, device=device)

            return layer
        else:
            raise TypeError(layer)

    def modify(self, layer: nn.Module, evolve_action: str) -> Tuple[nn.Linear, bool]:
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.LazyLinear):
            layer = copy.deepcopy(layer)

            in_features = layer.in_features
            out_features = layer.out_features
            weight_data = layer.weight.data
            bias_data = layer.bias.data if layer.bias.data is not None else None
            bias = False if layer.bias is None else True
            dtype = layer.weight.data.dtype
            device = layer.weight.data.device     
            done = False   
            

            if evolve_action == EvolveAction.IncreaseOut:
                new_layer = nn.Linear(in_features=in_features, out_features=out_features+1, bias=bias, device=device, dtype=dtype)

                new_weight = torch.empty(1, in_features, device=device)
                nn.init.kaiming_uniform_(new_weight, a=math.sqrt(5))

                new_bias = None
                if bias_data is not None:
                    new_bias = torch.empty(1, device=device)
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight_data)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(new_bias, -bound, bound)

                new_layer.weight =  nn.Parameter(torch.cat([weight_data, new_weight], dim=0))
                if new_layer.bias is not None:
                    new_layer.bias = nn.Parameter(torch.cat([bias_data, new_bias], dim=0))
                done = True

            elif evolve_action == EvolveAction.IncreaseIn:
                new_layer = nn.Linear(in_features=in_features+1, out_features=out_features, bias=bias, device=device, dtype=dtype)

                new_weight_column = torch.zeros((out_features, 1), device=device)
                new_layer.weight = nn.Parameter(torch.cat([weight_data, new_weight_column], dim=1))
                done = True

            elif evolve_action == EvolveAction.DecreaseOut:
                if out_features > 1:
                    new_layer = nn.Linear(in_features=in_features, out_features=out_features-1, bias=bias, device=device, dtype=dtype)
                
                    new_layer.weight = nn.Parameter(weight_data[:-1])
                    if new_layer.bias is not None:
                        new_layer.bias = nn.Parameter(bias_data[:-1])
                    done = True
                else:
                    new_layer = layer

            elif evolve_action == EvolveAction.DecreaseIn:
                if in_features > 1:
                    new_layer = nn.Linear(in_features=in_features-1, out_features=out_features, bias=bias, device=device, dtype=dtype)
                    new_layer.weight = nn.Parameter(weight_data[:, :-1])
                    done = True
                else:
                    new_layer = layer
                
            return new_layer, done
        else:
            raise TypeError(layer)


if __name__ == "__main__":
    # helper loss function
    def loss(x1:torch.Tensor, x2: torch.Tensor):
        val = torch.abs((x2/x1).mean() - 1) * 100
        print(f"loss = {val:.10f}%")

    linear = nn.Linear(in_features=3, out_features=2)
    print("Initial weights:")
    print(linear.weight)
    print("Initial bias:")
    print(linear.bias)

    linear, done = LinearModifier().modify(linear, EvolveAction.IncreaseOut)
    print("\nAfter adding a new output neuron:")
    print(linear.weight)
    print(linear.bias)

    linear, done = LinearModifier().modify(linear, EvolveAction.IncreaseIn)
    print("\nAfter increasing the input dimension:")
    print(linear.weight)
    print(linear.bias)

    identity_layer = LinearModifier().identity(linear)
    print("Weights after identity initialization:")
    print(identity_layer.weight)
    print("Bias after identity initialization:")
    print(identity_layer.bias)
