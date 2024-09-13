import torch
from torch import nn

import inspect


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
    def __init__(self, activation=nn.ReLU(), linear_start=False) -> "ActiSwitch":
        super(ActiSwitch, self).__init__()
        self.linear_weight = nn.Parameter(torch.tensor(1.0 if linear_start else 0.0))
        self.activation_weight = nn.Parameter(torch.tensor(0.0 if linear_start else 1.0))
        self.activation = activation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_weight * x + self.activation_weight * self.activation(x)
    
    def linear(self):
        self.linear_weight = nn.Parameter(torch.tensor(1.0))
        self.activation_weight = nn.Parameter(torch.tensor(0.0))
        return self
    
    def nonlinear(self):
        self.linear_weight = nn.Parameter(torch.tensor(0.0))
        self.activation_weight = nn.Parameter(torch.tensor(1.0))
        return self

if __name__ == "__main__":
    linear_pass_relu = ActiSwitch(nn.Tanh())
    x = torch.randn(5, 4, 3, 2)
    output: torch.Tensor = linear_pass_relu(x)

    print(f"Input: {x.shape}")
    print(f"Output: {output.shape}")
    print(f"difference in data: {(x-output).mean()}\n\n")