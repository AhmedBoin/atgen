import math
import torch
from torch import nn
import torch.nn.functional as F

from utils import conv2d_output_size


class Conv2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int=1, padding: int=1, bias: bool=True):
        super(Conv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.conv2d(input, self.weight, self.bias, self.stride, self.padding)
    
    def store_sizes(self, input_size):
        self.input_size = input_size
        self.output_size = (
            conv2d_output_size(input_size[0], self.kernel_size, self.padding, self.stride),
            conv2d_output_size(input_size[1], self.kernel_size, self.padding, self.stride),
        )

    def add_output_channel(self):
        new_weight = torch.empty(1, self.in_channels, self.kernel_size, self.kernel_size, device=self.weight.device)
        nn.init.kaiming_uniform_(new_weight, a=math.sqrt(5))
        
        new_bias = None
        if self.bias is not None:
            new_bias = torch.empty(1, device=self.bias.device)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(new_bias, -bound, bound)
            
        self.weight = nn.Parameter(torch.cat([self.weight.data, new_weight], dim=0))
        if self.bias is not None:
            self.bias = nn.Parameter(torch.cat([self.bias.data, new_bias], dim=0))

        self.out_channels += 1

    def add_input_channel(self):
        new_weight_column = torch.zeros((self.out_channels, 1, self.kernel_size, self.kernel_size), device=self.weight.device)
        self.weight = nn.Parameter(torch.cat([self.weight.data, new_weight_column], dim=1))
        self.in_channels += 1

    def remove_output_channel(self, index: int):
        if index < 0 or index >= self.out_channels:
            raise ValueError("Index out of range for output channels")

        self.weight = nn.Parameter(torch.cat([self.weight.data[:index], self.weight.data[index + 1:]], dim=0))
        
        if self.bias is not None:
            self.bias = nn.Parameter(torch.cat([self.bias.data[:index], self.bias.data[index + 1:]], dim=0))
        
        self.out_channels -= 1
    
    def remove_input_channel(self, index: int):
        if index < 0 or index >= self.in_channels:
            raise ValueError("Index out of range for input channels")

        self.weight = nn.Parameter(torch.cat([self.weight.data[:, :index, :, :], self.weight.data[:, index + 1:, :, :]], dim=1))
        self.in_channels -= 1

    @classmethod
    def init_identity_layer(cls, channels: int, kernel_size: int, bias: bool=True):
        """
        Class method to create a Conv2D layer with identity-like initialization.
        
        Args:
            kernel_size (int): Number of input and output channels (must be the same).
            bias (bool): Whether to include a bias term.
        
        Returns:
            Conv2D: An instance of Conv2D initialized as an identity layer.
        """
        
        layer = cls(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=bias)
        layer.weight.data = torch.zeros_like(layer.weight.data)
        # (out_channels, in_channels, kernel_size, kernel_size)

        with torch.no_grad():
            for i in range(channels):
                layer.weight.data[i, i, kernel_size//2, kernel_size//2] = 1.0

        if bias:
            layer.bias.data = torch.zeros_like(layer.bias.data)

        return layer
    
    @property
    def params_count(self):
        return self.weight.data.numel() + (self.bias.data.numel() if self.bias is not None else 0)
    

if __name__ == "__main__":
    # helper loss function
    def loss(x1:torch.Tensor, x2: torch.Tensor):
        val = torch.abs((x2/x1).mean() - 1) * 100
        print(f"loss = {val:.10f}%")
        
    identity_layer = Conv2D.init_identity_layer(channels=3, kernel_size=3)
    print("Weights after identity initialization:")
    print(identity_layer.weight)
    print("Bias after identity initialization:")
    print(identity_layer.bias)

    x = torch.randn(64, 3, 28, 28)
    conv = Conv2D(3, 32, 3)
    output = conv(x)

    print(f"Input: {x.shape}")
    print(f"Output shape: {output.shape}")

    before_conv = Conv2D.init_identity_layer(3, 3)
    output2 = conv(before_conv(x))

    print(f"Input: {x.shape}")
    print(f"Output shape: {output2.shape}")
    loss(output, output2)

    after_conv = Conv2D.init_identity_layer(32, 3)
    output3 = after_conv(conv(before_conv(x)))
    print(f"Input: {x.shape}")
    print(f"Output shape: {output3.shape}")
    loss(output, output3)