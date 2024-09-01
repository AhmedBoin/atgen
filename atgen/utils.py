import math
import torch.nn as nn


activation_functions = [
    nn.ReLU(),
    nn.Sigmoid(),
    nn.Tanh(),
    nn.LeakyReLU(),
    nn.ELU(),
    nn.SELU(),
    nn.GELU(),
    nn.Softplus(),
    nn.Softsign(),
    nn.Hardtanh(),
    nn.ReLU6(),
    nn.CELU(),
    nn.Hardswish(),
    nn.Hardsigmoid(),
    nn.Mish(),
]

# ANSI escape codes for colors
PURPLE = "\033[95m"
BLUE = "\033[38;5;153m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
GRAY = "\033[90m"
BOLD = "\033[1m"
RESET_COLOR = "\033[0m"


def conv2d_output_size(input_size, kernel_size, padding, stride):
    """
    Calculate the output size (height or width) of a Conv2D layer.

    Parameters:
    - input_size: int, the size of the input (height or width)
    - kernel_size: int, the size of the convolution kernel (height or width)
    - padding: int, the size of the padding added to each side
    - stride: int, the stride of the convolution

    Returns:
    - int, the output size after applying the Conv2D layer
    """
    return math.floor((input_size + 2 * padding - kernel_size) / stride) + 1

def calculate_linear_input_features(input_height, input_width, out_channels, kernel_size, padding, stride):
    """
    Calculate the number of input features for a linear layer after a Conv2D layer.

    Parameters:
    - input_height: int, the height of the input to the Conv2D layer
    - input_width: int, the width of the input to the Conv2D layer
    - out_channels: int, the number of output channels of the Conv2D layer
    - kernel_size: int, the size of the convolution kernel (assuming square kernel)
    - padding: int, the size of the padding added to each side
    - stride: int, the stride of the convolution

    Returns:
    - int, the number of input features for the subsequent linear layer
    """
    output_height = conv2d_output_size(input_height, kernel_size, padding, stride)
    output_width = conv2d_output_size(input_width, kernel_size, padding, stride)
    return out_channels * output_height * output_width