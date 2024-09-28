
import math
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torch.nn import init
import numpy as np


from .layers import (
    ActiSwitch, LayerModifier, LinearModifier, Conv1dModifier, Conv2dModifier, Conv3dModifier, 
    ConvTranspose1dModifier, ConvTranspose2dModifier, ConvTranspose3dModifier,
    BatchNorm1dModifier, BatchNorm2dModifier, BatchNorm3dModifier, InstanceNorm1dModifier, 
    InstanceNorm2dModifier, InstanceNorm3dModifier, LayerNormModifier, GroupNormModifier
    )


copy = {
    ActiSwitch: None,
    nn.ReLU: None,
    nn.Sigmoid: None,
    nn.Tanh: None,
    nn.LeakyReLU: None,
    nn.ELU: None,
    nn.SELU: None,
    nn.GELU: None,
    nn.Softplus: None,
    nn.Softsign: None,
    nn.Hardtanh: None,
    nn.ReLU6: None,
    nn.CELU: None,
    nn.Hardswish: None,
    nn.Hardsigmoid: None,
    nn.Mish: None,
    nn.LogSigmoid: None,
    nn.Softmax: None,
    nn.LogSoftmax: None,
    nn.Softmin: None,
    nn.Tanhshrink: None,
    nn.Softshrink: None,
    nn.Hardshrink: None,
    nn.GLU: None,
    nn.RReLU: None,
    nn.Threshold: None,
    nn.Dropout: None,
    nn.Dropout1d: None,
    nn.Dropout2d: None,
    nn.Dropout3d: None,
}

skip = {
    nn.Flatten: None,
    nn.MaxPool1d: None,
    nn.MaxPool2d: None,
    nn.MaxPool3d: None,
    nn.AvgPool1d: None,
    nn.AvgPool2d: None,
    nn.AvgPool3d: None,
    nn.AdaptiveMaxPool1d: None,
    nn.AdaptiveMaxPool2d: None,
    nn.AdaptiveMaxPool3d: None,
    nn.AdaptiveAvgPool1d: None,
    nn.AdaptiveAvgPool2d: None,
    nn.AdaptiveAvgPool3d: None,
}

follow = {
    nn.BatchNorm1d: BatchNorm1dModifier([], copy, skip),
    nn.BatchNorm2d: BatchNorm2dModifier([], copy, skip),
    nn.BatchNorm3d: BatchNorm3dModifier([], copy, skip),
    nn.LazyBatchNorm1d: BatchNorm1dModifier([], copy, skip),
    nn.LazyBatchNorm2d: BatchNorm2dModifier([], copy, skip),
    nn.LazyBatchNorm3d: BatchNorm3dModifier([], copy, skip),
    nn.InstanceNorm1d: InstanceNorm1dModifier([], copy, skip),
    nn.InstanceNorm2d: InstanceNorm2dModifier([], copy, skip),
    nn.InstanceNorm3d: InstanceNorm3dModifier([], copy, skip),
    nn.LayerNorm: LayerNormModifier([], copy, skip),
    nn.GroupNorm: GroupNormModifier([], copy, skip),
}

evolve: Dict[nn.Module, LayerModifier] = {
    nn.Linear: LinearModifier(follow, copy, skip),
    nn.LazyLinear: LinearModifier(follow, copy, skip),
    nn.Conv1d: Conv1dModifier(follow, copy, skip),
    nn.Conv2d: Conv2dModifier(follow, copy, skip),
    nn.Conv3d: Conv3dModifier(follow, copy, skip),
    nn.LazyConv1d: Conv1dModifier(follow, copy, skip),
    nn.LazyConv2d: Conv2dModifier(follow, copy, skip),
    nn.LazyConv3d: Conv3dModifier(follow, copy, skip),
    nn.ConvTranspose1d: ConvTranspose1dModifier(follow, copy, skip),
    nn.ConvTranspose2d: ConvTranspose2dModifier(follow, copy, skip),
    nn.ConvTranspose3d: ConvTranspose3dModifier(follow, copy, skip),
    nn.LazyConvTranspose1d: ConvTranspose1dModifier(follow, copy, skip),
    nn.LazyConvTranspose2d: ConvTranspose2dModifier(follow, copy, skip),
    nn.LazyConvTranspose3d: ConvTranspose3dModifier(follow, copy, skip),

    nn.Bilinear: None,
    nn.RNN: None,
    nn.GRU: None,
    nn.LSTM: None,
    nn.Embedding: None,
    nn.EmbeddingBag: None,
    nn.Transformer: None,
    nn.TransformerEncoder: None,
    nn.TransformerDecoder: None,
    nn.MultiheadAttention: None,
}



# ANSI escape codes(colors)
PURPLE = "\033[95m"
BLUE = "\033[38;5;153m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
GRAY = "\033[90m"
BOLD = "\033[1m"
RESET_COLOR = "\033[0m"


# Function to reset weights of the model
def reset_weights(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()

def custom_reset_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)  # Xavier initialization for weights
        if m.bias is not None:
            init.zeros_(m.bias)  # Zero initialization for biases


from tabulate import tabulate

def print_stats_table(best, metrics, fitness, population, species, config):
    met = ["Maximum", "Mean", "Minimum"]
    headers = ["Best", "Metrics", "Maximum", "Mean", "Minimum"]
    table = [[best, met[metrics], fitness[0], fitness[1], fitness[2]], 
             ["Crossover", "Mutation", "Perturbation", "Population", "Species"],
             [config.crossover_rate, config.mutation_rate, config.perturbation_rate, population, species]]
    print(tabulate(table, headers, tablefmt="fancy_grid"))



def merge_dicts(dict1: Dict[int, List[Tuple[nn.Sequential, float]]], dict2: Dict[int, List[Tuple[nn.Sequential, float]]]):
    for key, value in dict2.items():
        if key in dict1:
            dict1[key].extend(value)
        else:
            dict1[key] = value
    return dict1

def shift_to_positive(arr):
    min_val = min(arr)
    if min_val < 0:
        shift = abs(min_val) + 1  # Add a buffer of 1 to avoid zero
        return [x + shift for x in arr]
    return arr

def log_level(val, level):
    for _ in range(level):
        val = math.log(val+math.e-1)
    return val


def stable_softmax(x: List[float]) -> np.ndarray:
    x = np.array(x)
    return x - np.min(x)
    # exps = np.exp(x - np.max(x))
    # return exps / np.sum(exps)


"""
CrossOver Types
"""
def single_point_crossover(dna: nn.Sequential, rna: nn.Sequential):
    crossover_point = len(list(dna.parameters())) // 2
    for i, (param1, param2) in enumerate(zip(dna.parameters(), rna.parameters())):
        if i > crossover_point:
            param1.data = param2.data.clone()

def two_point_crossover(dna: nn.Sequential, rna: nn.Sequential):
    num_params = len(list(dna.parameters()))
    point1 = num_params // 3
    point2 = 2 * num_params // 3
    
    for i, (param1, param2) in enumerate(zip(dna.parameters(), rna.parameters())):
        if point1 < i < point2:
            param1.data = param2.data.clone()

def uniform_crossover(dna: nn.Sequential, rna: nn.Sequential):
    for param1, param2 in zip(dna.parameters(), rna.parameters()):
        mask = torch.rand_like(param1.data) > 0.5
        param1.data = torch.where(mask, param1.data, param2.data)

def arithmetic_crossover(dna: nn.Sequential, rna: nn.Sequential, alpha=0.5):
    for param1, param2 in zip(dna.parameters(), rna.parameters()):
        param1.data = alpha * param1.data + (1 - alpha) * param2.data

def blend_crossover(dna: nn.Sequential, rna: nn.Sequential, alpha=0.5):
    for param1, param2 in zip(dna.parameters(), rna.parameters()):
        min_val = torch.min(param1.data, param2.data)
        max_val = torch.max(param1.data, param2.data)
        diff = max_val - min_val
        param1.data = (min_val - alpha * diff) + torch.rand_like(param1.data) * (1 + 2 * alpha) * diff

def n_point_crossover(dna: nn.Sequential, rna: nn.Sequential, n=3):
    num_params = len(list(dna.parameters()))
    crossover_points = sorted([torch.randint(0, num_params, (1,)).item() for _ in range(n)])
    
    swap = False
    for i, (param1, param2) in enumerate(zip(dna.parameters(), rna.parameters())):
        if i in crossover_points:
            swap = not swap
        if swap:
            param1.data = param2.data.clone()

def hux_crossover(dna: nn.Sequential, rna: nn.Sequential):
    for param1, param2 in zip(dna.parameters(), rna.parameters()):
        mask = torch.rand_like(param1.data) > 0.5
        param1.data = torch.where(mask, param1.data, param2.data)
        param2.data = torch.where(mask, param2.data, param1.data)

def order_crossover(dna: nn.Sequential, rna: nn.Sequential):
    num_params = len(list(dna.parameters()))
    point1, point2 = sorted(torch.randint(0, num_params, (2,)).tolist())

    child_params = [None] * num_params
    child_params[point1:point2] = list(dna.parameters())[point1:point2]
    
    remaining_genes = [param for param in rna.parameters() if param not in child_params]
    for i in range(num_params):
        if child_params[i] is None:
            child_params[i] = remaining_genes.pop(0)
    
    for i, param in enumerate(dna.parameters()):
        param.data = child_params[i].data.clone()

import torch
import random

def pmx_crossover(dna: nn.Sequential, rna: nn.Sequential):
    """
    Perform Partially Mapped Crossover (PMX) on two neural networks (dna and rna).
    This operation swaps a segment of weights between the two networks and ensures
    a consistent mapping between the swapped segments.

    Parameters:
    - dna: The first parent network (PyTorch model)
    - rna: The second parent network (PyTorch model)

    The function modifies the `dna` network in place to represent the child.
    """

    # Get the list of parameters (weights) from both networks
    dna_params = list(dna.parameters())
    rna_params = list(rna.parameters())

    num_params = len(dna_params)

    # Select two random crossover points
    point1, point2 = sorted(random.sample(range(num_params), 2))

    # Initialize a mapping dictionary for the PMX operation
    mapping = {}

    # Perform PMX within the selected segment
    for i in range(point1, point2 + 1):
        dna_value = dna_params[i].clone()
        rna_value = rna_params[i].clone()

        # Swap the parameter values between dna and rna within the segment
        dna_params[i].data, rna_params[i].data = rna_value.data, dna_value.data

        # Update the mapping for the PMX operation
        mapping[dna_value.item()] = rna_value.item()

    # Fix the mapping outside of the crossover segment
    for i in range(num_params):
        if i < point1 or i > point2:
            original_value = dna_params[i].item()
            while original_value in mapping:
                original_value = mapping[original_value]

            dna_params[i].data.fill_(original_value)

def hyper_crossover(dna: nn.Sequential, rna: nn.Sequential):
    for param1, param2 in zip(dna.parameters(), rna.parameters()):
        mask = torch.rand_like(param1.data) > 0.5
        param1.data, param2.data = torch.where(mask, param1.data, param2.data), torch.where(mask, param2.data, param1.data)


crossover_dict = {
    'single_point': single_point_crossover,
    'two_point': two_point_crossover,
    'uniform': uniform_crossover, # prefer to use
    'arithmetic': arithmetic_crossover,
    'blend': blend_crossover, # prefer to use
    'npoint': n_point_crossover,
    'hux': hux_crossover,
    'order': order_crossover, # prefer not to use
    'pmx': pmx_crossover, # prefer not to use
    'hyper': hyper_crossover,
}


"""
Mutation
"""
def gaussian_mutation(param: torch.nn.Parameter, mutation_rate: float, perturbation_rate: float):
    mask = torch.rand_like(param) < mutation_rate
    noise = torch.randn_like(param) * perturbation_rate
    param.data.add_(mask * noise)

def uniform_mutation(param: torch.nn.Parameter, mutation_rate: float):
    mask = torch.rand_like(param) < mutation_rate
    param.data[mask] = torch.rand_like(param[mask])  # Replace with new random values

def swap_mutation(param: torch.nn.Parameter, mutation_rate: float):
    if torch.rand(1).item() < mutation_rate:
        idx1, idx2 = torch.randint(0, param.size(0), (2,))
        param[idx1], param[idx2] = param[idx2].clone(), param[idx1].clone()  # Swap rows

def scramble_mutation(param: torch.nn.Parameter, mutation_rate: float):
    if torch.rand(1).item() < mutation_rate:
        indices = torch.randperm(param.size(0))
        param.data.copy_(param[indices])  # Shuffle rows

def inversion_mutation(param: torch.nn.Parameter, mutation_rate: float):
    if torch.rand(1).item() < mutation_rate:
        start, end = sorted(torch.randint(0, param.size(0), (2,)).tolist())
        param[start:end] = param[start:end][::-1]  # Reverse the segment