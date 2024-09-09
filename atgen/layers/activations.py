import torch
from torch import nn

import inspect


class Pass(nn.Module):
    """pass input as it's, helper class for forward method of evolve network"""
    def forward(self, x: torch.Tensor):
        return x
    
    def print_layer(self, i: int):
        print(f"{self.__class__.__name__:<15}")

    # @property
    # def params_count(self):
    #     return 0


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
    def __init__(self, activation=nn.ReLU(), linear_start=False) -> "ActiSwitch":
        super(ActiSwitch, self).__init__()
        self.linear_weight = nn.Parameter(torch.tensor(1.0 if linear_start else 0.0))
        self.activation_weight = nn.Parameter(torch.tensor(0.0 if linear_start else 1.0))
        self.activation = activation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_weight * x + self.activation_weight * self.activation(x)
    
    def change_activation(self, activation=nn.ReLU):
        self.activation = activation()
    
    def print_layer(self, i: int):
        print(f"{self.__class__.__name__}({self.activation.__name__ if inspect.isfunction(self.activation) else self.activation.__class__.__name__}, {f'{100*(abs(self.activation_weight.item())/(abs(self.linear_weight.item())+abs(self.activation_weight.item()))):.2f}':<6}%)")

    # @property
    # def params_count(self):
    #     return 0


if __name__ == "__main__":
    linear_pass_relu = ActiSwitch(nn.Tanh())
    x = torch.randn(5, 4, 3, 2)
    output: torch.Tensor = linear_pass_relu(x)

    print(f"Input: {x.shape}")
    print(f"Output: {output.shape}")
    print(f"difference in data: {(x-output).mean()}\n\n")