import copy
from typing import List, Tuple
import torch
from torch import nn

from .utils import EvolveAction, LayerModifier

class BatchNorm1dModifier(LayerModifier):
    def __init__(self, follow: List[nn.Module] = [], copy: List[nn.Module] = [], skip: List[nn.Module] = []):
        super().__init__(follow, copy, skip)

    def identity(self, layer: nn.Module) -> nn.BatchNorm1d:
        if isinstance(layer, nn.BatchNorm1d):
            layer = copy.deepcopy(layer)

            # Reset running mean and variance to identity (mean = 0, variance = 1)
            layer.running_mean.data.zero_()
            layer.running_var.data.fill_(1)

            # Set weight to 1 and bias to 0 (for identity transformation)
            if layer.affine:
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

            return layer
        else:
            raise TypeError(layer)

    def modify(self, layer: nn.Module, evolve_action: str) -> Tuple[nn.BatchNorm1d, bool]:
        if isinstance(layer, nn.BatchNorm1d):
            layer = copy.deepcopy(layer)

            num_features = layer.num_features
            dtype = layer.weight.data.dtype
            device = layer.weight.data.device
            done = False

            if evolve_action == EvolveAction.IncreaseOut:
                new_layer = nn.BatchNorm1d(num_features + 1, affine=layer.affine, dtype=dtype, device=device)

                # Expand the weight, bias, running_mean, and running_var
                new_layer.weight.data = torch.cat([layer.weight.data, torch.ones(1, dtype=dtype, device=device)])
                new_layer.bias.data = torch.cat([layer.bias.data, torch.zeros(1, dtype=dtype, device=device)])
                new_layer.running_mean.data = torch.cat([layer.running_mean.data, torch.zeros(1, dtype=dtype, device=device)])
                new_layer.running_var.data = torch.cat([layer.running_var.data, torch.ones(1, dtype=dtype, device=device)])
                done = True

            elif evolve_action == EvolveAction.DecreaseOut and num_features > 1:
                new_layer = nn.BatchNorm1d(num_features - 1, affine=layer.affine, dtype=dtype, device=device)

                # Shrink the weight, bias, running_mean, and running_var
                new_layer.weight.data = layer.weight.data[:-1]
                new_layer.bias.data = layer.bias.data[:-1]
                new_layer.running_mean.data = layer.running_mean.data[:-1]
                new_layer.running_var.data = layer.running_var.data[:-1]
                done = True

            else:
                new_layer = layer

            return new_layer, done
        else:
            raise TypeError(layer)
        

class BatchNorm2dModifier(LayerModifier):
    def __init__(self, follow: List[nn.Module] = [], copy: List[nn.Module] = [], skip: List[nn.Module] = []):
        super().__init__(follow, copy, skip)

    def identity(self, layer: nn.Module) -> nn.BatchNorm2d:
        if isinstance(layer, nn.BatchNorm2d):
            layer = copy.deepcopy(layer)

            # Reset running mean and variance to identity (mean = 0, variance = 1)
            layer.running_mean.data.zero_()
            layer.running_var.data.fill_(1)

            # Set weight to 1 and bias to 0 (for identity transformation)
            if layer.affine:
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

            return layer
        else:
            raise TypeError(layer)

    def modify(self, layer: nn.Module, evolve_action: str) -> Tuple[nn.BatchNorm2d, bool]:
        if isinstance(layer, nn.BatchNorm2d):
            layer = copy.deepcopy(layer)

            num_features = layer.num_features
            dtype = layer.weight.data.dtype
            device = layer.weight.data.device
            done = False

            if evolve_action == EvolveAction.IncreaseOut:
                new_layer = nn.BatchNorm2d(num_features + 1, affine=layer.affine, dtype=dtype, device=device)

                # Expand the weight, bias, running_mean, and running_var
                new_layer.weight.data = torch.cat([layer.weight.data, torch.ones(1, dtype=dtype, device=device)])
                new_layer.bias.data = torch.cat([layer.bias.data, torch.zeros(1, dtype=dtype, device=device)])
                new_layer.running_mean.data = torch.cat([layer.running_mean.data, torch.zeros(1, dtype=dtype, device=device)])
                new_layer.running_var.data = torch.cat([layer.running_var.data, torch.ones(1, dtype=dtype, device=device)])
                done = True

            elif evolve_action == EvolveAction.DecreaseOut and num_features > 1:
                new_layer = nn.BatchNorm2d(num_features - 1, affine=layer.affine, dtype=dtype, device=device)

                # Shrink the weight, bias, running_mean, and running_var
                new_layer.weight.data = layer.weight.data[:-1]
                new_layer.bias.data = layer.bias.data[:-1]
                new_layer.running_mean.data = layer.running_mean.data[:-1]
                new_layer.running_var.data = layer.running_var.data[:-1]
                done = True

            else:
                new_layer = layer

            return new_layer, done
        else:
            raise TypeError(layer)
        

class BatchNorm3dModifier(LayerModifier):
    def __init__(self, follow: List[nn.Module] = [], copy: List[nn.Module] = [], skip: List[nn.Module] = []):
        super().__init__(follow, copy, skip)

    def identity(self, layer: nn.Module) -> nn.BatchNorm3d:
        if isinstance(layer, nn.BatchNorm3d):
            layer = copy.deepcopy(layer)

            # Reset running mean and variance to identity (mean = 0, variance = 1)
            layer.running_mean.data.zero_()
            layer.running_var.data.fill_(1)

            # Set weight to 1 and bias to 0 (for identity transformation)
            if layer.affine:
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

            return layer
        else:
            raise TypeError(layer)

    def modify(self, layer: nn.Module, evolve_action: str) -> Tuple[nn.BatchNorm3d, bool]:
        if isinstance(layer, nn.BatchNorm3d):
            layer = copy.deepcopy(layer)

            num_features = layer.num_features
            dtype = layer.weight.data.dtype
            device = layer.weight.data.device
            done = False

            if evolve_action == EvolveAction.IncreaseOut:
                new_layer = nn.BatchNorm3d(num_features + 1, affine=layer.affine, dtype=dtype, device=device)

                # Expand the weight, bias, running_mean, and running_var
                new_layer.weight.data = torch.cat([layer.weight.data, torch.ones(1, dtype=dtype, device=device)])
                new_layer.bias.data = torch.cat([layer.bias.data, torch.zeros(1, dtype=dtype, device=device)])
                new_layer.running_mean.data = torch.cat([layer.running_mean.data, torch.zeros(1, dtype=dtype, device=device)])
                new_layer.running_var.data = torch.cat([layer.running_var.data, torch.ones(1, dtype=dtype, device=device)])
                done = True

            elif evolve_action == EvolveAction.DecreaseOut and num_features > 1:
                new_layer = nn.BatchNorm3d(num_features - 1, affine=layer.affine, dtype=dtype, device=device)

                # Shrink the weight, bias, running_mean, and running_var
                new_layer.weight.data = layer.weight.data[:-1]
                new_layer.bias.data = layer.bias.data[:-1]
                new_layer.running_mean.data = layer.running_mean.data[:-1]
                new_layer.running_var.data = layer.running_var.data[:-1]
                done = True

            else:
                new_layer = layer

            return new_layer, done
        else:
            raise TypeError(layer)
        
        
class InstanceNorm1dModifier(LayerModifier):
    def __init__(self, follow: List[nn.Module] = [], copy: List[nn.Module] = [], skip: List[nn.Module] = []):
        super().__init__(follow, copy, skip)

    def identity(self, layer: nn.Module) -> nn.InstanceNorm1d:
        if isinstance(layer, nn.InstanceNorm1d):
            layer = copy.deepcopy(layer)

            # Reset running mean and variance to identity (mean = 0, variance = 1)
            layer.running_mean.data.zero_()
            layer.running_var.data.fill_(1)

            # Set weight to 1 and bias to 0 (for identity transformation)
            if layer.affine:
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

            return layer
        else:
            raise TypeError(layer)

    def modify(self, layer: nn.Module, evolve_action: str) -> Tuple[nn.InstanceNorm1d, bool]:
        if isinstance(layer, nn.InstanceNorm1d):
            layer = copy.deepcopy(layer)

            num_features = layer.num_features
            dtype = layer.weight.data.dtype
            device = layer.weight.data.device
            done = False

            if evolve_action == EvolveAction.IncreaseOut:
                new_layer = nn.InstanceNorm1d(num_features + 1, affine=layer.affine, dtype=dtype, device=device)

                # Expand the weight, bias, running_mean, and running_var
                if layer.affine:
                    new_layer.weight.data = torch.cat([layer.weight.data, torch.ones(1, dtype=dtype, device=device)])
                    new_layer.bias.data = torch.cat([layer.bias.data, torch.zeros(1, dtype=dtype, device=device)])
                new_layer.running_mean.data = torch.cat([layer.running_mean.data, torch.zeros(1, dtype=dtype, device=device)])
                new_layer.running_var.data = torch.cat([layer.running_var.data, torch.ones(1, dtype=dtype, device=device)])
                done = True

            elif evolve_action == EvolveAction.DecreaseOut and num_features > 1:
                new_layer = nn.InstanceNorm1d(num_features - 1, affine=layer.affine, dtype=dtype, device=device)

                # Shrink the weight, bias, running_mean, and running_var
                if layer.affine:
                    new_layer.weight.data = layer.weight.data[:-1]
                    new_layer.bias.data = layer.bias.data[:-1]
                new_layer.running_mean.data = layer.running_mean.data[:-1]
                new_layer.running_var.data = layer.running_var.data[:-1]
                done = True

            else:
                new_layer = layer

            return new_layer, done
        else:
            raise TypeError(layer)
        

class InstanceNorm2dModifier(LayerModifier):
    def __init__(self, follow: List[nn.Module] = [], copy: List[nn.Module] = [], skip: List[nn.Module] = []):
        super().__init__(follow, copy, skip)

    def identity(self, layer: nn.Module) -> nn.InstanceNorm2d:
        if isinstance(layer, nn.InstanceNorm2d):
            layer = copy.deepcopy(layer)

            # Reset running mean and variance to identity (mean = 0, variance = 1)
            layer.running_mean.data.zero_()
            layer.running_var.data.fill_(1)

            # Set weight to 1 and bias to 0 (for identity transformation)
            if layer.affine:
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

            return layer
        else:
            raise TypeError(layer)

    def modify(self, layer: nn.Module, evolve_action: str) -> Tuple[nn.InstanceNorm2d, bool]:
        if isinstance(layer, nn.InstanceNorm2d):
            layer = copy.deepcopy(layer)

            num_features = layer.num_features
            dtype = layer.weight.data.dtype
            device = layer.weight.data.device
            done = False

            if evolve_action == EvolveAction.IncreaseOut:
                new_layer = nn.InstanceNorm2d(num_features + 1, affine=layer.affine, dtype=dtype, device=device)

                # Expand the weight, bias, running_mean, and running_var
                if layer.affine:
                    new_layer.weight.data = torch.cat([layer.weight.data, torch.ones(1, dtype=dtype, device=device)])
                    new_layer.bias.data = torch.cat([layer.bias.data, torch.zeros(1, dtype=dtype, device=device)])
                new_layer.running_mean.data = torch.cat([layer.running_mean.data, torch.zeros(1, dtype=dtype, device=device)])
                new_layer.running_var.data = torch.cat([layer.running_var.data, torch.ones(1, dtype=dtype, device=device)])
                done = True

            elif evolve_action == EvolveAction.DecreaseOut and num_features > 1:
                new_layer = nn.InstanceNorm2d(num_features - 1, affine=layer.affine, dtype=dtype, device=device)

                # Shrink the weight, bias, running_mean, and running_var
                if layer.affine:
                    new_layer.weight.data = layer.weight.data[:-1]
                    new_layer.bias.data = layer.bias.data[:-1]
                new_layer.running_mean.data = layer.running_mean.data[:-1]
                new_layer.running_var.data = layer.running_var.data[:-1]
                done = True

            else:
                new_layer = layer

            return new_layer, done
        else:
            raise TypeError(layer)
        

class InstanceNorm3dModifier(LayerModifier):
    def __init__(self, follow: List[nn.Module] = [], copy: List[nn.Module] = [], skip: List[nn.Module] = []):
        super().__init__(follow, copy, skip)

    def identity(self, layer: nn.Module) -> nn.InstanceNorm3d:
        if isinstance(layer, nn.InstanceNorm3d):
            layer = copy.deepcopy(layer)

            # Reset running mean and variance to identity (mean = 0, variance = 1)
            layer.running_mean.data.zero_()
            layer.running_var.data.fill_(1)

            # Set weight to 1 and bias to 0 (for identity transformation)
            if layer.affine:
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

            return layer
        else:
            raise TypeError(layer)

    def modify(self, layer: nn.Module, evolve_action: str) -> Tuple[nn.InstanceNorm3d, bool]:
        if isinstance(layer, nn.InstanceNorm3d):
            layer = copy.deepcopy(layer)

            num_features = layer.num_features
            dtype = layer.weight.data.dtype
            device = layer.weight.data.device
            done = False

            if evolve_action == EvolveAction.IncreaseOut:
                new_layer = nn.InstanceNorm3d(num_features + 1, affine=layer.affine, dtype=dtype, device=device)

                # Expand the weight, bias, running_mean, and running_var
                if layer.affine:
                    new_layer.weight.data = torch.cat([layer.weight.data, torch.ones(1, dtype=dtype, device=device)])
                    new_layer.bias.data = torch.cat([layer.bias.data, torch.zeros(1, dtype=dtype, device=device)])
                new_layer.running_mean.data = torch.cat([layer.running_mean.data, torch.zeros(1, dtype=dtype, device=device)])
                new_layer.running_var.data = torch.cat([layer.running_var.data, torch.ones(1, dtype=dtype, device=device)])
                done = True

            elif evolve_action == EvolveAction.DecreaseOut and num_features > 1:
                new_layer = nn.InstanceNorm3d(num_features - 1, affine=layer.affine, dtype=dtype, device=device)

                # Shrink the weight, bias, running_mean, and running_var
                if layer.affine:
                    new_layer.weight.data = layer.weight.data[:-1]
                    new_layer.bias.data = layer.bias.data[:-1]
                new_layer.running_mean.data = layer.running_mean.data[:-1]
                new_layer.running_var.data = layer.running_var.data[:-1]
                done = True

            else:
                new_layer = layer

            return new_layer, done
        else:
            raise TypeError(layer)
        

class LayerNormModifier(LayerModifier):
    def __init__(self, follow: List[nn.Module] = [], copy: List[nn.Module] = [], skip: List[nn.Module] = []):
        super().__init__(follow, copy, skip)

    def identity(self, layer: nn.Module) -> nn.LayerNorm:
        if isinstance(layer, nn.LayerNorm):
            layer = copy.deepcopy(layer)

            # Set weight to 1 and bias to 0 (for identity transformation)
            if layer.elementwise_affine:
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

            return layer
        else:
            raise TypeError(layer)

    def modify(self, layer: nn.Module, evolve_action: str) -> Tuple[nn.LayerNorm, bool]:
        if isinstance(layer, nn.LayerNorm):
            layer = copy.deepcopy(layer)

            normalized_shape = list(layer.normalized_shape)
            dtype = layer.weight.data.dtype
            device = layer.weight.data.device
            done = False

            if evolve_action == EvolveAction.IncreaseOut:
                normalized_shape[0] += 1
                new_layer = nn.LayerNorm(normalized_shape, elementwise_affine=layer.elementwise_affine, dtype=dtype, device=device)

                # Expand the weight and bias
                if layer.elementwise_affine:
                    new_layer.weight.data = torch.cat([layer.weight.data, torch.ones(1, dtype=dtype, device=device)])
                    new_layer.bias.data = torch.cat([layer.bias.data, torch.zeros(1, dtype=dtype, device=device)])
                done = True

            elif evolve_action == EvolveAction.DecreaseOut and normalized_shape[0] > 1:
                normalized_shape[0] -= 1
                new_layer = nn.LayerNorm(normalized_shape, elementwise_affine=layer.elementwise_affine, dtype=dtype, device=device)

                # Shrink the weight and bias
                if layer.elementwise_affine:
                    new_layer.weight.data = layer.weight.data[:-1]
                    new_layer.bias.data = layer.bias.data[:-1]
                done = True

            else:
                new_layer = layer

            return new_layer, done
        else:
            raise TypeError(layer)
        

class GroupNormModifier(LayerModifier):
    def __init__(self, follow: List[nn.Module] = [], copy: List[nn.Module] = [], skip: List[nn.Module] = []):
        super().__init__(follow, copy, skip)

    def identity(self, layer: nn.Module) -> nn.GroupNorm:
        if isinstance(layer, nn.GroupNorm):
            layer = copy.deepcopy(layer)

            # Set weight to 1 and bias to 0 (for identity transformation)
            layer.weight.data.fill_(1)
            layer.bias.data.zero_()

            return layer
        else:
            raise TypeError(layer)

    def modify(self, layer: nn.Module, evolve_action: str) -> Tuple[nn.GroupNorm, bool]:
        if isinstance(layer, nn.GroupNorm):
            layer = copy.deepcopy(layer)

            num_groups = layer.num_groups
            num_channels = layer.num_channels
            dtype = layer.weight.data.dtype
            device = layer.weight.data.device
            done = False

            if evolve_action == EvolveAction.IncreaseOut:
                new_layer = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels + 1, affine=True, dtype=dtype, device=device)

                # Expand the weight and bias
                new_layer.weight.data = torch.cat([layer.weight.data, torch.ones(1, dtype=dtype, device=device)])
                new_layer.bias.data = torch.cat([layer.bias.data, torch.zeros(1, dtype=dtype, device=device)])
                done = True

            elif evolve_action == EvolveAction.DecreaseOut and num_channels > 1:
                new_layer = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels - 1, affine=True, dtype=dtype, device=device)

                # Shrink the weight and bias
                new_layer.weight.data = layer.weight.data[:-1]
                new_layer.bias.data = layer.bias.data[:-1]
                done = True

            else:
                new_layer = layer

            return new_layer, done
        else:
            raise TypeError(layer)
        
