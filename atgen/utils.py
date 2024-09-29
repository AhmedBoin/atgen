
import math
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import random



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
import torch
import random
import numpy as np

def flatten_params(model):
    # Flatten all parameters of the model into a 1D tensor
    return torch.cat([param.data.reshape(-1) for param in model.parameters()])

def rebuild_params(model, flat_params):
    # Convert flat_params back to the original shapes of the model's parameters
    param_start = 0
    for param in model.parameters():
        param_size = param.numel()
        param.data.copy_(flat_params[param_start:param_start + param_size].view(param.size()))
        param_start += param_size

def single_point_crossover(dna: nn.Sequential, rna: nn.Sequential):
    # Flatten parameters
    flat_dna = flatten_params(dna)
    flat_rna = flatten_params(rna)
    
    # Single-point crossover
    crossover_point = len(flat_dna) // 2
    child_flat = torch.cat((flat_dna[:crossover_point], flat_rna[crossover_point:]))
    
    # Rebuild child model
    rebuild_params(dna, child_flat)

def two_point_crossover(dna: nn.Sequential, rna: nn.Sequential):
    flat_dna = flatten_params(dna)
    flat_rna = flatten_params(rna)
    
    num_params = len(flat_dna)
    point1 = num_params // 3
    point2 = 2 * num_params // 3
    
    child_flat = torch.cat((flat_dna[:point1], flat_rna[point1:point2], flat_dna[point2:]))
    
    rebuild_params(dna, child_flat)

def uniform_crossover(dna: nn.Sequential, rna: nn.Sequential):
    flat_dna = flatten_params(dna)
    flat_rna = flatten_params(rna)
    
    mask = torch.rand_like(flat_dna) > 0.5
    child_flat = torch.where(mask, flat_dna, flat_rna)
    
    rebuild_params(dna, child_flat)

def arithmetic_crossover(dna: nn.Sequential, rna: nn.Sequential, alpha=0.5):
    flat_dna = flatten_params(dna)
    flat_rna = flatten_params(rna)
    
    child_flat = alpha * flat_dna + (1 - alpha) * flat_rna
    
    rebuild_params(dna, child_flat)

def blend_crossover(dna: nn.Sequential, rna: nn.Sequential, alpha=0.5):
    flat_dna = flatten_params(dna)
    flat_rna = flatten_params(rna)
    
    min_val = torch.min(flat_dna, flat_rna)
    max_val = torch.max(flat_dna, flat_rna)
    diff = max_val - min_val
    child_flat = (min_val - alpha * diff) + torch.rand_like(flat_dna) * (1 + 2 * alpha) * diff
    
    rebuild_params(dna, child_flat)

def n_point_crossover(dna: nn.Sequential, rna: nn.Sequential, n=3):
    flat_dna = flatten_params(dna)
    flat_rna = flatten_params(rna)
    
    num_params = len(flat_dna)
    crossover_points = sorted([torch.randint(0, num_params, (1,)).item() for _ in range(n)])
    
    swap = False
    child_flat = flat_dna.clone()
    for i in range(num_params):
        if i in crossover_points:
            swap = not swap
        if swap:
            child_flat[i] = flat_rna[i]
    
    rebuild_params(dna, child_flat)

def hux_crossover(dna: nn.Sequential, rna: nn.Sequential):
    flat_dna = flatten_params(dna)
    flat_rna = flatten_params(rna)
    
    mask = torch.rand_like(flat_dna) > 0.5
    child_flat = torch.where(mask, flat_dna, flat_rna)
    
    rebuild_params(dna, child_flat)

def order_crossover(dna: nn.Sequential, rna: nn.Sequential):
    flat_dna = flatten_params(dna)
    flat_rna = flatten_params(rna)
    
    num_params = len(flat_dna)
    point1, point2 = sorted(torch.randint(0, num_params, (2,)).tolist())

    child_flat = flat_dna.clone()
    child_flat[point1:point2] = flat_rna[point1:point2]
    
    remaining_genes = [gene for gene in flat_rna if gene not in child_flat[point1:point2]]
    for i in range(num_params):
        if i < point1 or i >= point2:
            try: child_flat[i] = remaining_genes.pop(0)
            except: pass
    
    rebuild_params(dna, child_flat)

def pmx_crossover(dna: nn.Sequential, rna: nn.Sequential):
    flat_dna = flatten_params(dna)
    flat_rna = flatten_params(rna)

    num_params = len(flat_dna)
    point1, point2 = sorted(random.sample(range(num_params), 2))
    
    # Initialize mapping
    mapping = {}
    
    # Perform PMX within the selected segment
    child_flat = flat_dna.clone()
    for i in range(point1, point2 + 1):
        mapping[flat_dna[i].item()] = flat_rna[i].item()
        child_flat[i] = flat_rna[i]
    
    # Fix the mapping outside of the crossover segment
    for i in range(num_params):
        if i < point1 or i > point2:
            original_value = flat_dna[i].item()
            while original_value in mapping:
                original_value = mapping[original_value]
            child_flat[i] = torch.tensor(original_value)
    
    rebuild_params(dna, child_flat)


crossover_dict = {
    'single_point': single_point_crossover,
    'two_point': two_point_crossover,
    'uniform': uniform_crossover, 
    'arithmetic': arithmetic_crossover, # prefer to use
    'blend': blend_crossover, # prefer to use
    'npoint': n_point_crossover,
    'hux': hux_crossover,
    'order': order_crossover, # prefer to use
    'pmx': pmx_crossover,
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
        param.data[start:end] = torch.flip(param.data[start:end], dims=[0])  # Reverse the segment