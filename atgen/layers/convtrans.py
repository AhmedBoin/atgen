import copy
import torch
import torch.nn as nn

import math
from typing import List, Tuple

from .utils import EvolveAction, LayerModifier


class ConvTranspose1dModifier(LayerModifier):
    def __init__(self, follow: List[nn.Module]=[], copy: List[nn.Module]=[], skip: List[nn.Module]=[]):
        super().__init__(follow, copy, skip)

    def new(self, layer: nn.Module) -> nn.ConvTranspose1d:
        if isinstance(layer, nn.ConvTranspose1d):
            layer = copy.deepcopy(layer)
            return nn.ConvTranspose1d(
                in_channels=layer.in_channels,
                out_channels=layer.out_channels,
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                output_padding=layer.output_padding,
                dilation=layer.dilation,
                groups=layer.groups,
                bias=False if layer.bias is None else True,
                dtype=layer.weight.data.dtype,
                device=layer.weight.data.device
            )

    def identity(self, layer: nn.Module) -> nn.ConvTranspose1d:
        if isinstance(layer, nn.ConvTranspose1d):
            layer = copy.deepcopy(layer)

            in_channels = layer.in_channels
            kernel_size = layer.kernel_size[0]
            stride = 1
            padding = kernel_size // 2
            output_padding = 0
            dilation = layer.dilation
            bias = False if layer.bias is None else True
            dtype = layer.weight.data.dtype
            device = layer.weight.data.device

            if kernel_size % 2 != 1:
                raise Exception("Kernel size must be an odd number to instantiate identity filter")

            layer = nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                dilation=dilation,
                bias=bias,
                dtype=dtype,
                device=device
            )

            with torch.no_grad():
                layer.weight.data = torch.zeros_like(layer.weight.data, dtype=dtype, device=device)
                center = kernel_size // 2
                for i in range(in_channels):
                    layer.weight.data[i, i, center] = 1.0
            if bias:
                layer.bias.data = torch.zeros_like(layer.bias.data, dtype=dtype, device=device)

            return layer
        else:
            raise TypeError(layer)

    def modify(self, layer: nn.Module, evolve_action: str) -> Tuple[nn.ConvTranspose1d, bool]:
        if isinstance(layer, nn.ConvTranspose1d):
            layer = copy.deepcopy(layer)

            in_channels = layer.in_channels
            out_channels = layer.out_channels
            kernel_size = layer.kernel_size
            stride = layer.stride
            padding = layer.padding
            output_padding = layer.output_padding
            dilation = layer.dilation
            bias = False if layer.bias is None else True
            dtype = layer.weight.data.dtype
            device = layer.weight.data.device
            done = False

            weight_data = layer.weight.data
            bias_data = layer.bias.data if layer.bias is not None else None

            if evolve_action == EvolveAction.IncreaseOut:
                new_out_channels = out_channels + 1
                new_layer = nn.ConvTranspose1d(
                    in_channels=in_channels,
                    out_channels=new_out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                    dilation=dilation,
                    bias=bias,
                    dtype=dtype,
                    device=device
                )

                new_weight = torch.empty((in_channels, 1, *kernel_size), device=device)
                nn.init.kaiming_uniform_(new_weight, a=math.sqrt(5))

                new_layer.weight = nn.Parameter(torch.cat([weight_data, new_weight[:, -1:, :]], dim=1))
                if bias:
                    new_bias = torch.zeros(1, device=device)
                    new_layer.bias = nn.Parameter(torch.cat([bias_data, new_bias]))
                done = True

            elif evolve_action == EvolveAction.IncreaseIn:
                new_in_channels = in_channels + 1
                new_layer = nn.ConvTranspose1d(
                    in_channels=new_in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                    dilation=dilation,
                    bias=bias,
                    dtype=dtype,
                    device=device
                )

                new_weight = torch.zeros((new_in_channels, out_channels, *kernel_size), device=device)
                new_weight[:in_channels, :, :] = weight_data
                new_layer.weight = nn.Parameter(new_weight)
                done = True

            elif evolve_action == EvolveAction.DecreaseOut:
                if out_channels > 1:
                    new_out_channels = out_channels - 1
                    new_layer = nn.ConvTranspose1d(
                        in_channels=in_channels,
                        out_channels=new_out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        output_padding=output_padding,
                        dilation=dilation,
                        bias=bias,
                        dtype=dtype,
                        device=device
                    )

                    new_layer.weight = nn.Parameter(weight_data[:, :-1, :])
                    if bias:
                        new_layer.bias = nn.Parameter(bias_data[:-1])
                    done = True
                else:
                    new_layer = layer

            elif evolve_action == EvolveAction.DecreaseIn:
                if in_channels > 1:
                    new_in_channels = in_channels - 1
                    new_layer = nn.ConvTranspose1d(
                        in_channels=new_in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        output_padding=output_padding,
                        dilation=dilation,
                        bias=bias,
                        dtype=dtype,
                        device=device
                    )

                    new_layer.weight = nn.Parameter(weight_data[:-1, :, :])
                    done = True
                else:
                    new_layer = layer

            return new_layer, done
        else:
            raise TypeError(layer)
        

class ConvTranspose2dModifier(LayerModifier):
    def __init__(self, follow: List[nn.Module]=[], copy: List[nn.Module]=[], skip: List[nn.Module]=[]):
        super().__init__(follow, copy, skip)

    def new(self, layer: nn.Module) -> nn.ConvTranspose2d:
        if isinstance(layer, nn.ConvTranspose2d):
            layer = copy.deepcopy(layer)
            return nn.ConvTranspose2d(
                in_channels=layer.in_channels,
                out_channels=layer.out_channels,
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                output_padding=layer.output_padding,
                dilation=layer.dilation,
                groups=layer.groups,
                bias=False if layer.bias is None else True,
                dtype=layer.weight.data.dtype,
                device=layer.weight.data.device
            )

    def identity(self, layer: nn.Module) -> nn.ConvTranspose2d:
        if isinstance(layer, nn.ConvTranspose2d):
            layer = copy.deepcopy(layer)

            in_channels = layer.in_channels
            kernel_size = layer.kernel_size
            stride = 1
            padding = layer.kernel_size[0]//2
            output_padding = 0
            dilation = layer.dilation
            bias = False if layer.bias is None else True
            dtype = layer.weight.data.dtype
            device = layer.weight.data.device

            if (kernel_size[0] % 2) != 1 or (kernel_size[1] % 2) != 1:
                raise Exception("Kernel size must be odd number in all dimensions to instantiate identity filter")

            layer = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                dilation=dilation,
                bias=bias,
                dtype=dtype,
                device=device
            )

            with torch.no_grad():
                layer.weight.data = torch.zeros_like(layer.weight.data, dtype=dtype, device=device)
                center = (kernel_size[0] // 2, kernel_size[1] // 2)
                for i in range(in_channels):
                    layer.weight.data[i, i, center[0], center[1]] = 1.0
            if bias:
                layer.bias.data = torch.zeros_like(layer.bias.data, dtype=dtype, device=device)

            return layer
        else:
            raise TypeError(layer)

    def modify(self, layer: nn.Module, evolve_action: str) -> Tuple[nn.ConvTranspose2d, bool]:
        if isinstance(layer, nn.ConvTranspose2d):
            layer = copy.deepcopy(layer)

            in_channels = layer.in_channels
            out_channels = layer.out_channels
            kernel_size = layer.kernel_size
            stride = layer.stride
            padding = layer.padding
            output_padding = layer.output_padding
            dilation = layer.dilation
            bias = False if layer.bias is None else True
            dtype = layer.weight.data.dtype
            device = layer.weight.data.device
            done = False

            weight_data = layer.weight.data
            bias_data = layer.bias.data if layer.bias is not None else None

            if evolve_action == EvolveAction.IncreaseOut:
                new_out_channels = out_channels + 1
                new_layer = nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=new_out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                    dilation=dilation,
                    bias=bias,
                    dtype=dtype,
                    device=device
                )

                new_weight = torch.empty((in_channels, 1, *kernel_size), device=device)
                nn.init.kaiming_uniform_(new_weight, a=math.sqrt(5))

                new_layer.weight = nn.Parameter(torch.cat([weight_data, new_weight[:, -1:, :, :]], dim=1))
                if bias:
                    new_bias = torch.zeros(1, device=device)
                    new_layer.bias = nn.Parameter(torch.cat([bias_data, new_bias]))
                done = True

            elif evolve_action == EvolveAction.IncreaseIn:
                new_in_channels = in_channels + 1
                new_layer = nn.ConvTranspose2d(
                    in_channels=new_in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                    dilation=dilation,
                    bias=bias,
                    dtype=dtype,
                    device=device
                )

                new_weight = torch.zeros((new_in_channels, out_channels, *kernel_size), device=device)
                new_weight[:in_channels, :, :, :] = weight_data
                new_layer.weight = nn.Parameter(new_weight)
                done = True

            elif evolve_action == EvolveAction.DecreaseOut:
                if out_channels > 1:
                    new_out_channels = out_channels - 1
                    new_layer = nn.ConvTranspose2d(
                        in_channels=in_channels,
                        out_channels=new_out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        output_padding=output_padding,
                        dilation=dilation,
                        bias=bias,
                        dtype=dtype,
                        device=device
                    )

                    new_layer.weight = nn.Parameter(weight_data[:, :-1, :, :])
                    if bias:
                        new_layer.bias = nn.Parameter(bias_data[:-1])
                    done = True
                else:
                    new_layer = layer

            elif evolve_action == EvolveAction.DecreaseIn:
                if in_channels > 1:
                    new_in_channels = in_channels - 1
                    new_layer = nn.ConvTranspose2d(
                        in_channels=new_in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        output_padding=output_padding,
                        dilation=dilation,
                        bias=bias,
                        dtype=dtype,
                        device=device
                    )

                    new_layer.weight = nn.Parameter(weight_data[:-1, :, :, :])
                    done = True
                else:
                    new_layer = layer

            return new_layer, done
        else:
            raise TypeError(layer)
        

class ConvTranspose3dModifier(LayerModifier):
    def __init__(self, follow: List[nn.Module]=[], copy: List[nn.Module]=[], skip: List[nn.Module]=[]):
        super().__init__(follow, copy, skip)

    def new(self, layer: nn.Module) -> nn.ConvTranspose3d:
        if isinstance(layer, nn.ConvTranspose3d):
            layer = copy.deepcopy(layer)
            return nn.ConvTranspose3d(
                in_channels=layer.in_channels,
                out_channels=layer.out_channels,
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                output_padding=layer.output_padding,
                dilation=layer.dilation,
                groups=layer.groups,
                bias=False if layer.bias is None else True,
                dtype=layer.weight.data.dtype,
                device=layer.weight.data.device
            )

    def identity(self, layer: nn.Module) -> nn.ConvTranspose3d:
        if isinstance(layer, nn.ConvTranspose3d):
            layer = copy.deepcopy(layer)

            in_channels = layer.in_channels
            kernel_size = layer.kernel_size
            stride = 1
            padding = [k // 2 for k in kernel_size]
            output_padding = 0
            dilation = layer.dilation
            bias = False if layer.bias is None else True
            dtype = layer.weight.data.dtype
            device = layer.weight.data.device

            if not all(k % 2 == 1 for k in kernel_size):
                raise Exception("Kernel size must be odd in all dimensions to instantiate identity filter")

            layer = nn.ConvTranspose3d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                dilation=dilation,
                bias=bias,
                dtype=dtype,
                device=device
            )

            with torch.no_grad():
                layer.weight.data = torch.zeros_like(layer.weight.data, dtype=dtype, device=device)
                center = tuple(k // 2 for k in kernel_size)
                for i in range(in_channels):
                    layer.weight.data[i, i, center[0], center[1], center[2]] = 1.0
            if bias:
                layer.bias.data = torch.zeros_like(layer.bias.data, dtype=dtype, device=device)

            return layer
        else:
            raise TypeError(layer)

    def modify(self, layer: nn.Module, evolve_action: str) -> Tuple[nn.ConvTranspose3d, bool]:
        if isinstance(layer, nn.ConvTranspose3d):
            layer = copy.deepcopy(layer)

            in_channels = layer.in_channels
            out_channels = layer.out_channels
            kernel_size = layer.kernel_size
            stride = layer.stride
            padding = layer.padding
            output_padding = layer.output_padding
            dilation = layer.dilation
            bias = False if layer.bias is None else True
            dtype = layer.weight.data.dtype
            device = layer.weight.data.device
            done = False

            weight_data = layer.weight.data
            bias_data = layer.bias.data if layer.bias is not None else None

            if evolve_action == EvolveAction.IncreaseOut:
                new_out_channels = out_channels + 1
                new_layer = nn.ConvTranspose3d(
                    in_channels=in_channels,
                    out_channels=new_out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                    dilation=dilation,
                    bias=bias,
                    dtype=dtype,
                    device=device
                )

                new_weight = torch.empty((in_channels, 1, *kernel_size), device=device)
                nn.init.kaiming_uniform_(new_weight, a=math.sqrt(5))

                new_layer.weight = nn.Parameter(torch.cat([weight_data, new_weight[:, -1:, :, :, :]], dim=1))
                if bias:
                    new_bias = torch.zeros(1, device=device)
                    new_layer.bias = nn.Parameter(torch.cat([bias_data, new_bias]))
                done = True

            elif evolve_action == EvolveAction.IncreaseIn:
                new_in_channels = in_channels + 1
                new_layer = nn.ConvTranspose3d(
                    in_channels=new_in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                    dilation=dilation,
                    bias=bias,
                    dtype=dtype,
                    device=device
                )

                new_weight = torch.zeros((new_in_channels, out_channels, *kernel_size), device=device)
                new_weight[:in_channels, :, :, :, :] = weight_data
                new_layer.weight = nn.Parameter(new_weight)
                done = True

            elif evolve_action == EvolveAction.DecreaseOut:
                if out_channels > 1:
                    new_out_channels = out_channels - 1
                    new_layer = nn.ConvTranspose3d(
                        in_channels=in_channels,
                        out_channels=new_out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        output_padding=output_padding,
                        dilation=dilation,
                        bias=bias,
                        dtype=dtype,
                        device=device
                    )

                    new_layer.weight = nn.Parameter(weight_data[:, :-1, :, :, :])
                    if bias:
                        new_layer.bias = nn.Parameter(bias_data[:-1])
                    done = True
                else:
                    new_layer = layer

            elif evolve_action == EvolveAction.DecreaseIn:
                if in_channels > 1:
                    new_in_channels = in_channels - 1
                    new_layer = nn.ConvTranspose3d(
                        in_channels=new_in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        output_padding=output_padding,
                        dilation=dilation,
                        bias=bias,
                        dtype=dtype,
                        device=device
                    )

                    new_layer.weight = nn.Parameter(weight_data[:-1, :, :, :, :])
                    done = True
                else:
                    new_layer = layer

            return new_layer, done
        else:
            raise TypeError(layer)
        
