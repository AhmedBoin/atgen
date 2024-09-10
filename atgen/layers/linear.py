import math
import torch
from torch import nn
import torch.nn.functional as F



class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
    def store_sizes(self, input_size, channels):
        self.input_size = input_size
        self.output_size: int = input_size[0] * input_size[1] * channels
        self.channels = channels
        return self.output_size, self.channels
    
    @property
    def params_count(self):
        """
        Get the count of parameters in the flatten layer (which is always zero).

        Returns:
            int: The number of parameters (always 0).
        """
        return 0
    
    def print_layer(self, i: int):
        print(f"Layer {i+1:<5}{self.__class__.__name__:<15}{(f'(batch_size, {self.output_size})'):<30}{self.params_count:<15}", end="")
    

    

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
        normalization (nn.Module or None): The normalization layer to apply after the linear transformation.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, norm_type: str = None):
        """
        Initializes the Linear layer with specified input and output features, and optionally a bias.

        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.
            bias (bool, optional): If `True`, a learnable bias vector is included in the layer. 
                                   Defaults to `True`.
            norm_type (str, optional): Type of normalization to use ('batch' or 'layer'). Defaults to `None`.
        """
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features)) if bias is not None else None
        self.norm_type = norm_type
        
        self.reset_parameters()

        # Initialize the normalization layer based on the norm_type parameter
        if norm_type == 'batch':
            self.normalization = nn.BatchNorm1d(out_features)
        elif norm_type == 'layer':
            self.normalization = nn.LayerNorm(out_features)
        else:
            self.normalization = None

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = F.linear(input, self.weight, self.bias)
        if self.normalization:
            output = self.normalization(output)
        return output

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

        # Update normalization layer if it exists
        if self.normalization:
            if isinstance(self.normalization, nn.BatchNorm1d):
                self.normalization = nn.BatchNorm1d(self.out_features)
            elif isinstance(self.normalization, nn.LayerNorm):
                self.normalization = nn.LayerNorm(self.out_features)

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

        # Update normalization layer if it exists
        if self.normalization:
            if isinstance(self.normalization, nn.BatchNorm1d):
                self.normalization = nn.BatchNorm1d(self.out_features)
            elif isinstance(self.normalization, nn.LayerNorm):
                self.normalization = nn.LayerNorm(self.out_features)

    def remove_weight(self, index: int):
        """Remove a specific input weight from each output neuron."""
        if index < 0 or index >= self.in_features:
            raise ValueError("Index out of range for input weights")

        self.weight = nn.Parameter(torch.cat([self.weight.data[:, :index], self.weight.data[:, index + 1:]], dim=1))
        self.in_features -= 1

    def reduce_dimensionality(self, method='variance'):
        """
        Reduces the dimensionality of the input by removing the least important input feature.

        Args:
            method (str): The method to use for determining feature importance ('variance' or 'correlation').
        """
        if method not in ['variance', 'correlation']:
            raise ValueError("Invalid method for dimensionality reduction. Choose 'variance' or 'correlation'.")

        with torch.no_grad():
            # Calculate feature importance based on the selected method
            if method == 'variance':
                feature_importances = torch.var(self.weight.data, dim=0)
            elif method == 'correlation':
                output = F.linear(torch.eye(self.in_features, device=self.weight.device), self.weight, self.bias)
                feature_importances = torch.abs(torch.corrcoef(output.T)[:, -1])

            # Find the index of the feature with the least importance
            least_important_feature_index = torch.argmin(feature_importances)

            # Remove the least important input weight
            self.remove_weight(least_important_feature_index)

            # Adjust remaining weights if necessary
            # For simplicity, let's renormalize the weights to maintain output scale
            self.weight.data *= (1 / (self.weight.data.norm(dim=1, keepdim=True) + 1e-6))

            # Update the normalization layer if it exists
            if self.normalization:
                if isinstance(self.normalization, nn.BatchNorm1d):
                    self.normalization = nn.BatchNorm1d(self.out_features)
                elif isinstance(self.normalization, nn.LayerNorm):
                    self.normalization = nn.LayerNorm(self.out_features)

    @classmethod
    def init_identity_layer(cls, size: int, bias: bool=True, norm_type: str = None):
        """
        Class method to create a Linear layer with identity-like initialization.
        
        Args:
            size (int): Number of input and output features (must be the same).
            bias (bool): Whether to include a bias term.
            norm_type (str): Type of normalization to use ('batch' or 'layer'). Defaults to `None`.
        
        Returns:
            Linear: An instance of Linear initialized as an identity layer.
        """
        layer = cls(in_features=size, out_features=size, bias=bias, norm_type=norm_type)
        layer.weight.data = torch.eye(size, size)

        if bias:
            layer.bias.data = torch.zeros_like(layer.bias.data)

        return layer
    
    @property
    def params_count(self):
        return self.weight.data.numel() + (self.bias.data.numel() if self.bias is not None else 0)
    
    def print_layer(self, i: int):
        print(f"Layer {i+1:<5}{self.__class__.__name__:<15}{(f'(batch_size, {self.out_features})'):<30}{self.params_count:<15}", end="")
    

class LazyLinear(Linear):
    def __init__(self, out_features: int, bias: bool = True, norm_type: str = None, input_size=None):
        if input_size is not None:
            super(LazyLinear, self).__init__(input_size, out_features, bias, norm_type)
        self.out_features = out_features
        self.bias = bias
        self.norm_type = norm_type
        
    def custom_init(self, input_size):
        super(LazyLinear, self).__init__(input_size, self.out_features, self.bias, self.norm_type)
        return self.out_features


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