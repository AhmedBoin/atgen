import math
import torch
from torch import nn
import torch.nn.functional as F


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
    

if __name__ == "__main__":
    # helper loss function
    def loss(x1:torch.Tensor, x2: torch.Tensor):
        val = torch.abs((x2/x1).mean() - 1) * 100
        print(f"loss = {val:.10f}%")

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