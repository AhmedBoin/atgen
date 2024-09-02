import math
import torch
from torch import nn
import torch.nn.functional as F

from .utils import conv2d_output_size


class Conv2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int=1, padding: int=1, bias: bool=True, norm: bool=False):
        super(Conv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.norm = norm  # Flag to indicate whether to apply normalization

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        # Add normalization layer if requested
        if self.norm:
            self.norm_layer = nn.BatchNorm2d(out_channels)
        else:
            self.norm_layer = None

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.input_size is None:
            self.store_sizes((input.size(2), input.size(3)))
        x = F.conv2d(input, self.weight, self.bias, self.stride, self.padding)
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        return x

    def store_sizes(self, input_size):
        self.input_size = input_size
        self.output_size = (
            conv2d_output_size(input_size[0], self.kernel_size, self.padding, self.stride),
            conv2d_output_size(input_size[1], self.kernel_size, self.padding, self.stride),
        )
        return self.output_size, self.out_channels

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
        
        # Update normalization layer if necessary
        if self.norm_layer is not None:
            self.norm_layer = nn.BatchNorm2d(self.out_channels)

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
        
        # Update normalization layer if necessary
        if self.norm_layer is not None:
            self.norm_layer = nn.BatchNorm2d(self.out_channels)

    def remove_input_channel(self, index: int):
        if index < 0 or index >= self.in_channels:
            raise ValueError("Index out of range for input channels")

        self.weight = nn.Parameter(torch.cat([self.weight.data[:, :index, :, :], self.weight.data[:, index + 1:, :, :]], dim=1))
        self.in_channels -= 1

    def change_kernel_size(self, kernel_size=None):
        new_kernel_size = kernel_size if kernel_size is not None else (self.kernel_size + 2)
        padding_size = (new_kernel_size - self.kernel_size) // 2

        self.kernel_size = new_kernel_size
        self.padding = self.kernel_size // 2
        
        new_weight = torch.zeros(
            (self.out_channels, self.in_channels, new_kernel_size, new_kernel_size), 
            device=self.weight.device
        )

        # Copy the existing weights into the center of the new weight tensor
        new_weight[:, :, padding_size:padding_size + self.weight.size(2), padding_size:padding_size + self.weight.size(3)] = self.weight.data
        self.weight = nn.Parameter(new_weight)

    @classmethod
    def init_identity_layer(cls, channels: int, kernel_size: int, bias: bool=True, norm: bool=False):
        """
        Class method to create a Conv2D layer with identity-like initialization.
        
        Args:
            kernel_size (int): Number of input and output channels (must be the same).
            bias (bool): Whether to include a bias term.
        
        Returns:
            Conv2D: An instance of Conv2D initialized as an identity layer.
        """
        
        layer = cls(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=bias, norm=norm)
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
    
    def print_layer(self, i: int):
        print(f"Layer {i+1:<5}{self.__class__.__name__:<15}{(f'(batch_size, {self.out_channels}, {self.output_size[0]}, {self.output_size[1]})'):<30}{self.params_count:<15}", end="")
    

class LazyConv2D(Conv2D):
    def __init__(self, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 1, bias: bool = True, norm: bool = False, in_channels: int=None):
        if in_channels is not None:
            super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias, norm)
        else:
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding
            self.bias = bias
            self.norm = norm

    def custom_init(self, in_channels: int):
        super().__init__(in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.bias, self.norm)
    

class MaxPool2D(nn.Module):
    def __init__(self, kernel_size: int = 2, stride: int = 2, padding: int = 0):
        """
        Custom MaxPool2D Layer that can be dynamically modified.

        Args:
            kernel_size (int): Size of the window to take a max over.
            stride (int, optional): Stride of the window. Default is 1.
            padding (int, optional): Implicit zero padding to be added on both sides. Default is 0.
        """
        super(MaxPool2D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.input_size = None
        self.output_size = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Perform max pooling on the input tensor.

        Args:
            input (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Output tensor after max pooling.
        """
        if self.input_size is None:
            self.store_sizes((input.size(2), input.size(3)))
        return F.max_pool2d(input, self.kernel_size, self.stride, self.padding)

    def store_sizes(self, input_size, channels):
        """
        Store the input and output sizes for future reference.

        Args:
            input_size (tuple): Size of the input tensor (height, width).
        """
        self.input_size = input_size
        self.output_size = (
            (input_size[0] - self.kernel_size + 2 * self.padding) // self.stride + 1,
            (input_size[1] - self.kernel_size + 2 * self.padding) // self.stride + 1
        )
        self.channels = channels
        return self.output_size, self.channels

    def update_kernel_size(self, new_kernel_size: int):
        """
        Update the kernel size of the max pooling layer and adjust the output size accordingly.

        Args:
            new_kernel_size (int): The new kernel size to set.
        """
        self.kernel_size = new_kernel_size
        if self.input_size is not None:
            self.store_sizes(self.input_size)

    def update_stride(self, new_stride: int):
        """
        Update the stride of the max pooling layer and adjust the output size accordingly.

        Args:
            new_stride (int): The new stride value to set.
        """
        self.stride = new_stride
        if self.input_size is not None:
            self.store_sizes(self.input_size)

    def update_padding(self, new_padding: int):
        """
        Update the padding of the max pooling layer and adjust the output size accordingly.

        Args:
            new_padding (int): The new padding value to set.
        """
        self.padding = new_padding
        if self.input_size is not None:
            self.store_sizes(self.input_size)

    @property
    def params_count(self):
        """
        Get the count of parameters in the max pooling layer (which is always zero for pooling layers).

        Returns:
            int: The number of parameters (always 0 for pooling layers).
        """
        return 0  # MaxPooling layers do not have learnable parameters
    
    def print_layer(self, i: int):
        print(f"Layer {i+1:<5}{self.__class__.__name__:<15}{(f'(batch_size, {self.channels}, {self.output_size[0]}, {self.output_size[1]})'):<30}{self.params_count:<15}", end="")


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



    # Initialize the custom MaxPool2D layer
    max_pool_layer = MaxPool2D(kernel_size=2, stride=2)

    # Example input tensor
    input_tensor = torch.randn(1, 1, 4, 4)

    # Perform max pooling
    output = max_pool_layer(input_tensor)
    print("Output after pooling:", output)
    print("Stored output size:", max_pool_layer.output_size)

    # Dynamically update kernel size and see changes in output size
    max_pool_layer.update_kernel_size(3)
    output = max_pool_layer(input_tensor)
    print("Output after updating kernel size:", output)
    print("Updated stored output size:", max_pool_layer.output_size)