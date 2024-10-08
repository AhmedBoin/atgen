import math
import torch
from torch import nn

from .utils import EvolveAction, LayerModifier

import copy
from typing import List, Tuple


class Conv1dModifier(LayerModifier):
    def __init__(self, follow: List[nn.Module]=[], copy: List[nn.Module]=[], skip: List[nn.Module]=[]):
        super().__init__(follow, copy, skip)

    def new(self, layer: nn.Module) -> nn.Conv1d:
        if isinstance(layer, nn.Conv1d):
            layer = copy.deepcopy(layer)

            return nn.Conv1d(
                in_channels=layer.in_channels,
                out_channels=layer.out_channels,
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                dilation=layer.dilation,
                groups=layer.groups,
                bias=False if layer.bias is None else True,
                dtype=layer.weight.data.dtype,
                device=layer.weight.data.device
            )

    def identity(self, layer: nn.Module) -> nn.Conv1d:
        if isinstance(layer, nn.Conv1d):
            layer = copy.deepcopy(layer)

            in_channels = layer.in_channels
            kernel_size = layer.kernel_size
            dilation = layer.dilation
            groups = layer.groups
            bias = False if layer.bias is None else True
            dtype = layer.weight.data.dtype
            device = layer.weight.data.device

            if kernel_size[0] % 2 != 1:
                raise Exception("Kernel size must be odd number to instantiate identity filter")

            layer = nn.Conv1d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size[0] // 2,
                dilation=dilation,
                groups=groups,
                bias=bias,
                dtype=dtype,
                device=device
            )

            with torch.no_grad():
                layer.weight.data = torch.zeros_like(layer.weight.data, dtype=dtype, device=device)
                for i in range(in_channels):
                    layer.weight.data[i, i, kernel_size[0] // 2] = 1.0
            if bias:
                layer.bias.data = torch.zeros_like(layer.bias.data, dtype=dtype, device=device)

            return layer
        else:
            raise TypeError(layer)

    def modify(self, layer: nn.Module, evolve_action: str) -> Tuple[nn.Conv1d, bool]:
        if isinstance(layer, nn.Conv1d):
            layer = copy.deepcopy(layer)

            in_channels = layer.in_channels
            out_channels = layer.out_channels
            kernel_size = layer.kernel_size
            stride = layer.stride
            padding = layer.padding
            dilation = layer.dilation
            groups = layer.groups
            bias = False if layer.bias is None else True
            dtype = layer.weight.data.dtype
            device = layer.weight.data.device
            done = False

            weight_data = layer.weight.data
            bias_data = layer.bias.data if layer.bias is not None else None     

            if evolve_action == EvolveAction.IncreaseOut:
                new_out_channels = out_channels + 1
                new_layer = nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=new_out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    bias=bias,
                    dtype=dtype,
                    device=device
                )

                new_weight = torch.empty((1, in_channels, kernel_size), device=device)
                nn.init.kaiming_uniform_(new_weight, a=math.sqrt(5))

                new_layer.weight = nn.Parameter(torch.cat([weight_data, new_weight[-1:, :, :]], dim=0))
                if bias:
                    new_bias = torch.zeros(1, device=device)
                    new_layer.bias = nn.Parameter(torch.cat([bias_data, new_bias]))
                done = True

            elif evolve_action == EvolveAction.IncreaseIn:
                new_in_channels = in_channels + 1
                new_layer = nn.Conv1d(
                    in_channels=new_in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    bias=bias,
                    dtype=dtype,
                    device=device
                )

                new_weight = torch.zeros((out_channels, new_in_channels, kernel_size), device=device)
                new_weight[:, :in_channels, :] = weight_data
                new_layer.weight = nn.Parameter(new_weight)
                done = True

            elif evolve_action == EvolveAction.DecreaseOut:
                if out_channels > 1:
                    new_out_channels = out_channels - 1
                    new_layer = nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=new_out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        groups=groups,
                        bias=bias,
                        dtype=dtype,
                        device=device
                    )

                    new_layer.weight = nn.Parameter(weight_data[:-1])
                    if bias:
                        new_layer.bias = nn.Parameter(bias_data[:-1])
                    done = True
                else:
                    new_layer = layer

            elif evolve_action == EvolveAction.DecreaseIn:
                if in_channels > 1:
                    new_in_channels = in_channels - 1
                    new_layer = nn.Conv1d(
                        in_channels=new_in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        groups=groups,
                        bias=bias,
                        dtype=dtype,
                        device=device
                    )

                    new_layer.weight = nn.Parameter(weight_data[:, :-1, :])
                    done = True
                else:
                    new_layer = layer

            return new_layer, done
        else:
            raise TypeError(layer)


class Conv2dModifier(LayerModifier):
    def __init__(self, follow: List[nn.Module]=[], copy: List[nn.Module]=[], skip: List[nn.Module]=[]):
        super().__init__(follow, copy, skip)

    def new(self, layer: nn.Module) -> nn.Conv2d:
        if isinstance(layer, nn.Conv2d):
            layer = copy.deepcopy(layer)

            return nn.Conv2d(
                in_channels=layer.in_channels,
                out_channels=layer.out_channels,
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                dilation=layer.dilation,
                groups=layer.groups,
                bias=False if layer.bias is None else True,
                dtype=layer.weight.data.dtype,
                device=layer.weight.data.device
            )

    def identity(self, layer: nn.Module) -> nn.Conv2d:
        if isinstance(layer, nn.Conv2d):
            layer = copy.deepcopy(layer)

            channels = layer.in_channels
            kernel_size = layer.kernel_size
            dilation = layer.dilation
            groups = layer.groups
            bias = False if layer.bias is None else True
            dtype = layer.weight.data.dtype
            device = layer.weight.data.device

            if (kernel_size[0] % 2) != 1:
                raise Exception("Kernel size must be odd number to instantiate identity filter")

            layer = nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size[0]//2, kernel_size[1]//2),
                dilation=dilation,
                groups=groups,
                bias=bias,
                dtype=dtype,
                device=device
            )

            with torch.no_grad():
                layer.weight.data = torch.zeros_like(layer.weight.data, dtype=dtype, device=device)
                for i in range(channels):
                    layer.weight.data[i, i, kernel_size[0]//2, kernel_size[1]//2] = 1.0
            if bias:
                layer.bias.data = torch.zeros_like(layer.bias.data, dtype=dtype, device=device)

            return layer
        else:
            raise TypeError(layer)

    def modify(self, layer: nn.Module, evolve_action: str) -> Tuple[nn.Conv2d, bool]:
        if isinstance(layer, nn.Conv2d):
            layer = copy.deepcopy(layer)

            in_channels = layer.in_channels
            out_channels = layer.out_channels
            kernel_size = layer.kernel_size
            stride = layer.stride
            padding = layer.padding
            dilation = layer.dilation
            groups = layer.groups
            bias = False if layer.bias is None else True
            dtype = layer.weight.data.dtype
            device = layer.weight.data.device
            done = False

            weight_data = layer.weight.data
            bias_data = layer.bias.data if layer.bias is not None else None     

            if evolve_action == EvolveAction.IncreaseOut:
                new_out_channels = out_channels + 1
                new_layer = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=new_out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    bias=bias,
                    dtype=dtype,
                    device=device
                )

                new_weight = torch.empty((1, in_channels, *kernel_size), device=device)
                nn.init.kaiming_uniform_(new_weight, a=math.sqrt(5))

                new_layer.weight = nn.Parameter(torch.cat([weight_data, new_weight], dim=0))
                if bias:
                    new_bias = torch.zeros(1, device=device)
                    new_layer.bias = nn.Parameter(torch.cat([bias_data, new_bias]))
                done = True

            elif evolve_action == EvolveAction.IncreaseIn:
                new_in_channels = in_channels + 1
                new_layer = nn.Conv2d(
                    in_channels=new_in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    bias=bias,
                    dtype=dtype,
                    device=device
                )

                new_weight = torch.zeros((out_channels, new_in_channels, *kernel_size), device=device)
                new_weight[:, :in_channels, :, :] = weight_data
                new_layer.weight = nn.Parameter(new_weight)
                done = True

            elif evolve_action == EvolveAction.DecreaseOut:
                if out_channels > 1:
                    new_out_channels = out_channels - 1
                    new_layer = nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=new_out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        groups=groups,
                        bias=bias,
                        dtype=dtype,
                        device=device
                    )

                    new_layer.weight = nn.Parameter(weight_data[:-1])
                    if bias:
                        new_layer.bias = nn.Parameter(bias_data[:-1])
                    done = True
                else:
                    new_layer = layer

            elif evolve_action == EvolveAction.DecreaseIn:
                if in_channels > 1:
                    new_in_channels = in_channels - 1
                    new_layer = nn.Conv2d(
                        in_channels=new_in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        groups=groups,
                        bias=bias,
                        dtype=dtype,
                        device=device
                    )

                    new_layer.weight = nn.Parameter(weight_data[:, :-1, :, :])
                    done = True
                else:
                    new_layer = layer

            return new_layer, done
        else:
            raise TypeError(layer)
        

class Conv3dModifier(LayerModifier):
    def __init__(self, follow: List[nn.Module]=[], copy: List[nn.Module]=[], skip: List[nn.Module]=[]):
        super().__init__(follow, copy, skip)

    def new(self, layer: nn.Module) -> nn.Conv3d:
        if isinstance(layer, nn.Conv3d):
            layer = copy.deepcopy(layer)

            return nn.Conv3d(
                in_channels=layer.in_channels,
                out_channels=layer.out_channels,
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                dilation=layer.dilation,
                groups=layer.groups,
                bias=False if layer.bias is None else True,
                dtype=layer.weight.data.dtype,
                device=layer.weight.data.device
            )


    def identity(self, layer: nn.Module) -> nn.Conv3d:
        if isinstance(layer, nn.Conv3d):
            layer = copy.deepcopy(layer)

            in_channels = layer.in_channels
            kernel_size = layer.kernel_size
            dilation = layer.dilation
            groups = layer.groups
            bias = False if layer.bias is None else True
            dtype = layer.weight.data.dtype
            device = layer.weight.data.device

            if (kernel_size[0] % 2) != 1 or (kernel_size[1] % 2) != 1 or (kernel_size[2] % 2) != 1:
                raise Exception("Kernel size must be odd number in all dimensions to instantiate identity filter")

            layer = nn.Conv3d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size[0]//2, kernel_size[1]//2, kernel_size[2]//2),
                dilation=dilation,
                groups=groups,
                bias=bias,
                dtype=dtype,
                device=device
            )

            with torch.no_grad():
                layer.weight.data = torch.zeros_like(layer.weight.data, dtype=dtype, device=device)
                center = (kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2)
                for i in range(in_channels):
                    layer.weight.data[i, i, center[0], center[1], center[2]] = 1.0
            if bias:
                layer.bias.data = torch.zeros_like(layer.bias.data, dtype=dtype, device=device)

            return layer
        else:
            raise TypeError(layer)

    def modify(self, layer: nn.Module, evolve_action: str) -> Tuple[nn.Conv3d, bool]:
        if isinstance(layer, nn.Conv3d):
            layer = copy.deepcopy(layer)

            in_channels = layer.in_channels
            out_channels = layer.out_channels
            kernel_size = layer.kernel_size
            stride = layer.stride
            padding = layer.padding
            dilation = layer.dilation
            groups = layer.groups
            bias = False if layer.bias is None else True
            dtype = layer.weight.data.dtype
            device = layer.weight.data.device
            done = False

            weight_data = layer.weight.data
            bias_data = layer.bias.data if layer.bias is not None else None

            if evolve_action == EvolveAction.IncreaseOut:
                new_out_channels = out_channels + 1
                new_layer = nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=new_out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    bias=bias,
                    dtype=dtype,
                    device=device
                )

                new_weight = torch.empty((1, in_channels, *kernel_size), device=device)
                nn.init.kaiming_uniform_(new_weight, a=math.sqrt(5))

                new_layer.weight = nn.Parameter(torch.cat([weight_data, new_weight[-1:, :, :, :, :]], dim=0))
                if bias:
                    new_bias = torch.zeros(1, device=device)
                    new_layer.bias = nn.Parameter(torch.cat([bias_data, new_bias]))
                done = True

            elif evolve_action == EvolveAction.IncreaseIn:
                new_in_channels = in_channels + 1
                new_layer = nn.Conv3d(
                    in_channels=new_in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    bias=bias,
                    dtype=dtype,
                    device=device
                )

                new_weight = torch.zeros((out_channels, new_in_channels, *kernel_size), device=device)
                new_weight[:, :in_channels, :, :, :] = weight_data
                new_layer.weight = nn.Parameter(new_weight)
                done = True

            elif evolve_action == EvolveAction.DecreaseOut:
                if out_channels > 1:
                    new_out_channels = out_channels - 1
                    new_layer = nn.Conv3d(
                        in_channels=in_channels,
                        out_channels=new_out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        groups=groups,
                        bias=bias,
                        dtype=dtype,
                        device=device
                    )

                    new_layer.weight = nn.Parameter(weight_data[:-1])
                    if bias:
                        new_layer.bias = nn.Parameter(bias_data[:-1])
                    done = True
                else:
                    new_layer = layer

            elif evolve_action == EvolveAction.DecreaseIn:
                if in_channels > 1:
                    new_in_channels = in_channels - 1
                    new_layer = nn.Conv3d(
                        in_channels=new_in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        groups=groups,
                        bias=bias,
                        dtype=dtype,
                        device=device
                    )

                    new_layer.weight = nn.Parameter(weight_data[:, :-1, :, :, :])
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
        
    layer = nn.Conv2d(3, 32, 3, 1, 1)
    identity_layer = Conv2dModifier().identity(layer)

    print("Weights after identity initialization:")
    print(identity_layer.weight)
    print("Bias after identity initialization:")
    print(identity_layer.bias)

    x = torch.randn(64, 3, 28, 28)
    output = layer(x)
    print(f"Input: {x.shape}")
    print(f"Output shape: {output.shape}")

    output2 = layer(identity_layer(x))
    print(f"Input: {x.shape}")
    print(f"Output shape: {output2.shape}")
    loss(output, output2)