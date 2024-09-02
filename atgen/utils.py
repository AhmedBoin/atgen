
import torch.nn as nn


activation_functions = [
    nn.ReLU,
    nn.Sigmoid,
    nn.Tanh,
    nn.LeakyReLU,
    nn.ELU,
    nn.SELU,
    nn.GELU,
    nn.Softplus,
    nn.Softsign,
    nn.Hardtanh,
    nn.ReLU6,
    nn.CELU,
    nn.Hardswish,
    nn.Hardsigmoid,
    nn.Mish,
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


from tabulate import tabulate

def print_stats_table(best, maximum, mean, minimum, population):
    headers = ["Best", "Maximum", "Mean", "Minimum", "Population"]
    table = [[best, maximum, mean, minimum, population]]
    print(tabulate(table, headers, tablefmt="fancy_grid"))


if __name__ == "__main__":
    # Example usage
    best_value = 0.85
    mean_value = 0.75
    min_value = 0.65

    print_stats_table(best_value, mean_value, min_value)
