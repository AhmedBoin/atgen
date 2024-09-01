import math
import torch
from torch import nn
import torch.nn.functional as F

from utils import conv2d_output_size


class Pass(nn.Module):
    """pass input as it's, helper class for forward method of evolve network"""
    def forward(self, x: torch.Tensor):
        return x


class GradientControl(torch.autograd.Function):
    """
    A custom autograd function to control the gradient flow for the weight parameter in the ActiSwitch layer.

    `GradientControl` is designed to ensure that the weight parameter within the `ActiSwitch` module 
    stays within the range [0, 1] during training. It allows the gradient to propagate in a controlled 
    manner, enforcing the following rules:
    
    - If the weight is between 0 and 1, the gradient is applied normally.
    - If the weight is exactly 0, only positive gradients are allowed to increase the weight.
    - If the weight is exactly 1, only negative gradients are allowed to decrease the weight.

    This function modifies the gradient during backpropagation to ensure the weight does not 
    exceed the bounds of [0, 1] while still allowing learning when at the boundaries.

    Methods:
        forward(ctx, weight: torch.Tensor) -> torch.Tensor:
            Saves the input tensor for use in the backward pass and returns the tensor unchanged.
        backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
            Adjusts the gradient based on the weight's current value:
              - Passes the gradient through if the weight is within bounds.
              - Allows increasing gradient only if the weight is at 0.
              - Allows decreasing gradient only if the weight is at 1.
    """
    @staticmethod
    def forward(ctx, weight):
        ctx.save_for_backward(weight)
        return torch.sigmoid(weight * 20)

    @staticmethod
    def backward(ctx, grad_output):
        (weight,) = ctx.saved_tensors
        
        return torch.sigmoid(weight) * (1-torch.sigmoid(weight)) * grad_output


class ActiSwitch(nn.Module):
    """
    A custom neural network module that allows dynamic blending between a linear function 
    and a non-linear activation function based on two learnable weighting factors.

    The `ActiSwitch` module is designed to facilitate a smooth transition between a linear 
    input-output relationship and a non-linear transformation. This is achieved by two learnable 
    parameters (`linear_weight` and `activation_weight`) that control the blending ratio between 
    the linear pass-through and the non-linear activation function. The sum of these weights 
    is constrained to 1, ensuring a balanced combination during training. The parameters adjust 
    dynamically, and the module provides flexibility in choosing different activation functions 
    as needed.

    Attributes:
        linear_weight (torch.nn.Parameter): A learnable parameter initialized based on the `backprob_phase`.
                                            It controls the contribution of the linear pass-through in the output.
        activation_weight (torch.nn.Parameter): A learnable parameter initialized to complement `linear_weight` 
                                                ensuring their sum is 1. It controls the contribution of the 
                                                non-linear activation function in the output.
        activation (callable): An activation function (e.g., `nn.ReLU()` or `nn.Tanh()`) that defines the 
                               non-linear transformation applied to the input.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Computes the forward pass by combining the linear and non-linear outputs based on 
            their respective weights.
        change_activation(activation: callable) -> None:
            Updates the activation function used for the non-linear transformation.
    """
    def __init__(self, activation=nn.ReLU(), backprob_phase=True) -> "ActiSwitch":
        super(ActiSwitch, self).__init__()
        self.linear_weight = nn.Parameter(torch.tensor(1.0 if backprob_phase else 0.0))
        self.activation_weight = nn.Parameter(torch.tensor(0.0 if backprob_phase else 1.0))
        self.activation = activation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_weight * x + self.activation_weight * self.activation(x)
    
    def change_activation(self, activation=nn.ReLU()):
        self.activation = activation
    

class Linear(nn.Module):
    """
    A custom fully connected linear layer that allows dynamic addition and removal of neurons 
    (output features) and weights (input features). This layer is an extension of the traditional 
    linear layer (`torch.nn.Linear`) with more flexibility to modify the network architecture 
    during training or model evolution.

    Attributes:
        in_features (int): The number of input features for each neuron in this layer.
        out_features (int): The number of output neurons in this layer.
        weight (torch.nn.Parameter): The learnable weight matrix of shape (out_features, in_features).
        bias (torch.nn.Parameter or None): The learnable bias vector of shape (out_features,). 
                                           If `bias` is set to `False`, no bias vector will be used.

    Methods:
        reset_parameters() -> None:
            Initializes the weights and biases of the layer using Kaiming uniform initialization.
        
        forward(input: torch.Tensor) -> torch.Tensor:
            Performs the forward pass by applying a linear transformation to the input tensor.

        add_neuron() -> None:
            Adds a new output neuron to the layer, initializing its weights and bias to small values.
        
        add_weight() -> None:
            Increases the input dimension by adding a new input weight to each output neuron, initialized to zero.
        
        remove_neuron(index: int) -> None:
            Removes a specific output neuron from the layer based on the given index.
        
        remove_weight(index: int) -> None:
            Removes a specific input weight from each output neuron based on the given index.
        
        init_identity_layer(size: int, bias: bool=True) -> Linear:
            Class method to create a Linear layer with identity-like initialization. 
            Useful for initializing a layer to behave as an identity function for square matrices.
        
        params_count() -> int:
            Returns the total number of learnable parameters in the layer, including both weights and biases.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        Initializes the Linear layer with specified input and output features, and optionally a bias.

        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.
            bias (bool, optional): If `True`, a learnable bias vector is included in the layer. 
                                   Defaults to `True`.
        """
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
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
        return F.linear(input, self.weight, self.bias)

    def add_neuron(self):
        """Add a new output neuron with weights initialized to a small value."""
        new_weight = torch.empty(1, self.in_features, device=self.weight.device)
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

        self.out_features += 1

    def add_weight(self):
        """Increase the input dimension by adding a new input weight to each output neuron with a small value."""
        new_weight_column = torch.zeros((self.out_features, 1), device=self.weight.device)
        self.weight = nn.Parameter(torch.cat([self.weight.data, new_weight_column], dim=1))
        self.in_features += 1

    def remove_neuron(self, index: int):
        """Remove a specific output neuron from the layer."""
        if index < 0 or index >= self.out_features:
            raise ValueError("Index out of range for output neurons")

        self.weight = nn.Parameter(torch.cat([self.weight.data[:index], self.weight.data[index + 1:]], dim=0))
        
        if self.bias is not None:
            self.bias = nn.Parameter(torch.cat([self.bias.data[:index], self.bias.data[index + 1:]], dim=0))
        
        self.out_features -= 1

    def remove_weight(self, index: int):
        """Remove a specific input weight from each output neuron."""
        if index < 0 or index >= self.in_features:
            raise ValueError("Index out of range for input weights")

        self.weight = nn.Parameter(torch.cat([self.weight.data[:, :index], self.weight.data[:, index + 1:]], dim=1))
        self.in_features -= 1

    @classmethod
    def init_identity_layer(cls, size: int, bias: bool=True):
        """
        Class method to create a Linear layer with identity-like initialization.
        
        Args:
            size (int): Number of input and output features (must be the same).
            bias (bool): Whether to include a bias term.
        
        Returns:
            Linear: An instance of Linear initialized as an identity layer.
        """
        
        layer = cls(in_features=size, out_features=size, bias=bias)
        layer.weight.data = torch.zeros_like(layer.weight.data)

        with torch.no_grad():
            for i in range(size):
                layer.weight.data[i, i] = 1.0

        if bias:
            layer.bias.data = torch.zeros_like(layer.bias.data)

        return layer
    
    @property
    def params_count(self):
        return self.weight.data.numel() + (self.bias.data.numel() if self.bias is not None else 0)
    

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

    linear_pass_relu = ActiSwitch(nn.Tanh())
    x = torch.randn(5, 4, 3, 2)
    output: torch.Tensor = linear_pass_relu(x)

    print(f"Input: {x.shape}")
    print(f"Output: {output.shape}")
    print(f"difference in data: {(x-output).mean()}\n\n")

    linear = Linear(in_features=3, out_features=2)
    print("Initial weights:")
    print(linear.weight)
    print("Initial bias:")
    print(linear.bias)

    linear.add_neuron()
    print("\nAfter adding a new output neuron:")
    print(linear.weight)
    print(linear.bias)

    linear.add_weight()
    print("\nAfter increasing the input dimension:")
    print(linear.weight)
    print(linear.bias)

    identity_layer = Linear.init_identity_layer(size=3)
    print("Weights after identity initialization:")
    print(identity_layer.weight)
    print("Bias after identity initialization:")
    print(identity_layer.bias)

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






