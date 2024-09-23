
import math
from typing import Dict, List, Tuple
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
    # exps = np.exp(x - np.max(x))
    # return exps / np.sum(exps)
    return x - np.min(x) + 1

