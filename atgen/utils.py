
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

def print_stats_table(best, metrics, maximum, mean, minimum, population, species=1):
    met = ["Maximum", "Mean", "Minimum"]
    headers = ["Best", "Metrics","Maximum", "Mean", "Minimum", "Population", "Species"]
    table = [[best, met[metrics], maximum, mean, minimum, population, species]]
    print(tabulate(table, headers, tablefmt="fancy_grid"))


