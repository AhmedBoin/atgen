import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def store_sizes(self, input_size):
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

    def params_count(self):
        """
        Get the count of parameters in the max pooling layer (which is always zero for pooling layers).

        Returns:
            int: The number of parameters (always 0 for pooling layers).
        """
        return 0  # MaxPooling layers do not have learnable parameters
    

if __name__ == "__main__":

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