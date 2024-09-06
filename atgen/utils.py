
import torch.nn as nn
from .layers import Linear, Conv2D, MaxPool2D, Flatten, LazyLinear


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


from typing import List, Tuple, Union

class GenomeType:
    def __init__(self, layers: List[Union['Linear', 'Conv2D', 'MaxPool2D', 'Flatten']]):
        """
        Initialize the GenomeType with a list of layers. Each layer is represented by its type
        and relevant information (e.g., features for Linear, kernel size for Conv2D).
        """
        self.genome = []
        self.split_indices = []
        self.parse_layers(layers)
    
    def parse_layers(self, layers: List[Union['Linear', 'Conv2D', 'MaxPool2D', 'Flatten']]):
        """
        Parse the layers to extract their type and critical information.
        """
        for i, layer in enumerate(layers):
            if isinstance(layer, Linear) or isinstance(layer, LazyLinear):
                self.genome.append(('Linear', layer.in_features, layer.out_features))
            elif isinstance(layer, Conv2D):
                self.genome.append(('Conv2D', layer.in_channels, layer.out_channels, layer.kernel_size))
            elif isinstance(layer, MaxPool2D):
                self.genome.append(('MaxPool2D', layer.kernel_size))
                self.split_indices.append(i)  # Mark index for splitting during crossover
            elif isinstance(layer, Flatten):
                self.genome.append(('Flatten',))
                self.split_indices.append(i)  # Mark index for splitting during crossover
    
    def crossover(self, other_genome: 'GenomeType') -> Tuple['GenomeType', 'GenomeType']:
        """
        Perform crossover between two GenomeTypes by splitting at critical layers (MaxPool2D and Flatten)
        and recombining the sections.
        """
        child1_genome = []
        child2_genome = []

        # Get the split sections
        sections_self = self.split_into_sections()
        sections_other = other_genome.split_into_sections()

        # Perform crossover at the section level
        for sec_self, sec_other in zip(sections_self, sections_other):
            # Randomly swap sections (you can use a random function here for variety)
            child1_genome.extend(sec_self)
            child2_genome.extend(sec_other)
        
        return GenomeType(child1_genome), GenomeType(child2_genome)

    def split_into_sections(self) -> List[List[Tuple]]:
        """
        Split the genome into sections based on the indices of critical layers (MaxPool2D, Flatten).
        """
        sections = []
        start = 0
        for index in self.split_indices:
            sections.append(self.genome[start:index])
            start = index + 1
        sections.append(self.genome[start:])  # Add the remaining part
        return sections

    def __repr__(self):
        """
        String representation of the genome for easy debugging.
        """
        return f"GenomeType({self.genome})"


# Example usage:
# Assuming `Linear`, `Conv2D`, `MaxPool2D`, and `Flatten` are previously defined layer classes
layer_list = [
    Linear(in_features=128, out_features=64),
    Conv2D(in_channels=3, out_channels=16, kernel_size=(3, 3)),
    MaxPool2D(kernel_size=(2, 2)),
    Flatten(),
    Linear(in_features=64, out_features=10)
]

# Create two genome types
genome1 = GenomeType(layer_list)
genome2 = GenomeType(layer_list)

# Perform crossover
child1, child2 = genome1.crossover(genome2)

print(child1)
print(child2)



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
