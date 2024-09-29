import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
from typing import *

from atgen.layers.utils import EvolveAction, LayerModifier


class SplineLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)

class RadialBasisFunction(nn.Module):
    def __init__(
        self,
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        denominator: float = None,  # larger denominators lead to smoother basis
    ):
        super().__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)

class FastKANLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        use_base_update: bool = True,
        use_layernorm: bool = True,
        base_activation = F.silu,
        spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layernorm = None
        if use_layernorm:
            assert input_dim > 1, "Do not use layernorms on 1D inputs. Set `use_layernorm=False`."
            self.layernorm = nn.LayerNorm(input_dim)
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, spline_weight_init_scale)
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, use_layernorm=True):
        if self.layernorm is not None and use_layernorm:
            spline_basis = self.rbf(self.layernorm(x))
        else:
            spline_basis = self.rbf(x)
        ret = self.spline_linear(spline_basis.view(*spline_basis.shape[:-2], -1))
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base
        return ret

    def plot_curve(
        self,
        input_index: int,
        output_index: int,
        num_pts: int = 1000,
        num_extrapolate_bins: int = 2
    ):
        '''this function returns the learned curves in a FastKANLayer.
        input_index: the selected index of the input, in [0, input_dim) .
        output_index: the selected index of the output, in [0, output_dim) .
        num_pts: num of points sampled for the curve.
        num_extrapolate_bins (N_e): num of bins extrapolating from the given grids. The curve 
            will be calculate in the range of [grid_min - h * N_e, grid_max + h * N_e].
        '''
        ng = self.rbf.num_grids
        h = self.rbf.denominator
        assert input_index < self.input_dim
        assert output_index < self.output_dim
        w = self.spline_linear.weight[
            output_index, input_index * ng : (input_index + 1) * ng
        ]   # num_grids,
        x = torch.linspace(
            self.rbf.grid_min - num_extrapolate_bins * h,
            self.rbf.grid_max + num_extrapolate_bins * h,
            num_pts
        )   # num_pts, num_grids
        with torch.no_grad():
            y = (w * self.rbf(x.to(w.dtype))).sum(-1)
        return x, y
    

class FastKANLayerModifier(LayerModifier):
    def __init__(self, follow: List[nn.Module]=[], copy: List[nn.Module]=[], skip: List[nn.Module]=[]):
        super().__init__(follow, copy, skip)

    def new(self, layer: nn.Module) -> nn.Module:
        if isinstance(layer, FastKANLayer):
            layer = copy.deepcopy(layer)
            return FastKANLayer(
                input_dim=layer.input_dim,
                output_dim=layer.output_dim,
                grid_min=layer.rbf.grid_min,
                grid_max=layer.rbf.grid_max,
                num_grids=layer.rbf.num_grids,
                use_base_update=layer.use_base_update,
                use_layernorm=layer.layernorm is not None,
                base_activation=layer.base_activation,
                spline_weight_init_scale=layer.spline_linear.init_scale,
            )
        
    def identity(self, layer: nn.Module) -> nn.Module:
        if isinstance(layer, FastKANLayer):
            layer = copy.deepcopy(layer)
            
            # Initialize the BaseLinear layer as identity if present
            if layer.use_base_update and layer.base_linear:
                input_dim = layer.input_dim
                base_linear = nn.Linear(input_dim, input_dim, bias=False)
                base_linear.weight.data = torch.eye(input_dim)
                layer.base_linear = base_linear
            
            # Set the SplineLinear to approximate identity behavior (neutral effect)
            output_dim = layer.output_dim
            input_dim = layer.input_dim * layer.rbf.num_grids
            spline_linear = SplineLinear(input_dim * layer.rbf.num_grids, output_dim, init_scale=0)
            spline_linear.weight.data.fill_(0)  # Neutral spline weights
            layer.spline_linear = spline_linear
            
            return layer
        else:
            raise TypeError(f"Expected FastKANLayer, got {type(layer)}")

    def modify(self, layer: nn.Module, evolve_action: str) -> Tuple[nn.Module, bool]:
        if isinstance(layer, FastKANLayer):
            layer = copy.deepcopy(layer)

            input_dim = layer.input_dim
            output_dim = layer.output_dim
            grid_min = layer.rbf.grid_min
            grid_max = layer.rbf.grid_max
            num_grids = layer.rbf.num_grids
            use_layernorm = layer.layernorm is not None
            spline_weight_init_scale = layer.spline_linear.init_scale
            base_activation = layer.base_activation
            use_base_update = layer.use_base_update
            done = False

            if evolve_action == EvolveAction.IncreaseOut:
                new_layer = FastKANLayer(
                    input_dim=input_dim,
                    output_dim=output_dim + 1,
                    grid_min=grid_min,
                    grid_max=grid_max,
                    num_grids=num_grids,
                    use_base_update=use_base_update,
                    use_layernorm=use_layernorm,
                    base_activation=base_activation,
                    spline_weight_init_scale=spline_weight_init_scale,
                )
                done = True

            elif evolve_action == EvolveAction.IncreaseIn:
                new_layer = FastKANLayer(
                    input_dim=input_dim + 1,
                    output_dim=output_dim,
                    grid_min=grid_min,
                    grid_max=grid_max,
                    num_grids=num_grids,
                    use_base_update=use_base_update,
                    use_layernorm=use_layernorm,
                    base_activation=base_activation,
                    spline_weight_init_scale=spline_weight_init_scale,
                )
                done = True

            elif evolve_action == EvolveAction.DecreaseOut:
                if output_dim > 1:
                    new_layer = FastKANLayer(
                        input_dim=input_dim,
                        output_dim=output_dim - 1,
                        grid_min=grid_min,
                        grid_max=grid_max,
                        num_grids=num_grids,
                        use_base_update=use_base_update,
                        use_layernorm=use_layernorm,
                        base_activation=base_activation,
                        spline_weight_init_scale=spline_weight_init_scale,
                    )
                    done = True
                else:
                    new_layer = layer

            elif evolve_action == EvolveAction.DecreaseIn:
                if input_dim > 1:
                    new_layer = FastKANLayer(
                        input_dim=input_dim - 1,
                        output_dim=output_dim,
                        grid_min=grid_min,
                        grid_max=grid_max,
                        num_grids=num_grids,
                        use_base_update=use_base_update,
                        use_layernorm=use_layernorm,
                        base_activation=base_activation,
                        spline_weight_init_scale=spline_weight_init_scale,
                    )
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

    linear = FastKANLayer(3, 2)
    print("Initial weights:")
    print(linear.weight)
    print("Initial bias:")
    print(linear.bias)

    linear, done = FastKANLayerModifier().modify(linear, EvolveAction.IncreaseOut)
    print("\nAfter adding a new output neuron:")
    print(linear.weight)
    print(linear.bias)

    linear, done = FastKANLayerModifier().modify(linear, EvolveAction.IncreaseIn)
    print("\nAfter increasing the input dimension:")
    print(linear.weight)
    print(linear.bias)

    identity_layer = FastKANLayerModifier().identity(linear)
    print("Weights after identity initialization:")
    print(identity_layer.weight)
    print("Bias after identity initialization:")
    print(identity_layer.bias)