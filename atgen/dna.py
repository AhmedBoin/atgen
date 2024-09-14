import copy
import random
import inspect
from typing import List, Union

import torch
from torch import nn

from utils import BLUE, BOLD, RESET_COLOR
from layers import ActiSwitch, EvolveAction
from config import ATGENConfig


class EvolveBlock:
    '''
    This class represents a block of neural network layers that can evolve. 
    Each block consists of multiple layers, and provides methods to mutate, 
    evolve, and create identity transformations of the layers within the block.
    
    Attributes:
    -----------
    - modules (List[nn.Module]): List of layers in the block.
    - config (ATGENConfig): Configuration object containing evolution settings.
    
    Methods:
    --------
    - append(item): Adds a new layer to the block.
    - pop(idx=-1): Removes the last or specified layer from the block.
    - full(): Returns True if the block contains at least one layer.
    - genes(): Calculates the total number of parameters in the block.
    - mutate(): Checks if the first layer in the block can be mutated (requires gradient).
    - __iter__(): Returns an iterator over the block's layers.
    - __len__(): Returns the number of layers in the block.
    - evolve(action): Evolves the block based on a specific action (e.g., increase or decrease layer size).
    - identity(): Returns a new block with identity transformations applied to the layers.
    '''
    def __init__(self, module: List[nn.Module], config: ATGENConfig):
        self.modules = module
        self.config = config

    def append(self, item):
        self.modules.append(item)

    def pop(self, idx=-1):
        self.modules.pop(idx)

    def full(self):
        return len(self.modules) != 0
    
    def genes(self):
        return sum([param.numel() for module in self.modules for param in module.parameters()])
    
    def mutate(self):
        for param in self.modules[0].parameters():
            return param.requires_grad
        
    def __iter__(self):
        return iter(self.modules)
    
    def __len__(self):
        return len(self.modules)
    
    def structure(self):
        kind = self.modules[0].__class__.__name__
        if kind.startswith("Lazy"):
            kind = kind[4:]
        return kind
    
    def new(self) -> List[nn.Module]:
        new = []
        modifier = self.config.evolve[self.modules[0].__class__]
        new.append(copy.deepcopy(modifier.new(self.modules[0])))
        new.extend(copy.deepcopy(self.modules[1:]))
        return new
    
    def size(self) -> int:
        modifier = self.config.evolve[self.modules[0].__class__]
        layer, _ = modifier.modify(self.modules[0], EvolveAction.IncreaseOut)
        after = sum([param.numel() for param in layer.parameters()])
        before = sum([param.numel() for param in self.modules[0].parameters()])
        return int(before/(after-before))

    def evolve(self, action: str) -> Union[bool, int]:
        evolved = []
        conversion: int = None

        # main layer
        modifier = self.config.evolve[self.modules[0].__class__]
        layer, done = modifier.modify(self.modules[0], action)
        evolved.append(layer)

        # follower layer
        for layer in self.modules[1:]:
            if layer.__class__ in modifier.follow:
                f_modifier = self.config.follow[layer.__class__]
                n_layer, _ = f_modifier.modify(layer, action)
                evolved.append(n_layer)
            else:
                evolved.append(layer)

            if isinstance(layer, nn.Flatten): # special case
                if isinstance(self.modules[0], nn.Conv2d):
                    conversion: int = self.modules[0].weight.data.shape[0]

        self.modules = evolved
        return conversion if conversion is not None else done
            
    
    def identity(self) -> "EvolveBlock":
        evolved = []

        # main layer
        modifier = self.config.evolve[self.modules[0].__class__]
        layer = modifier.identity(self.modules[0])
        evolved.append(layer)

        # follower layer
        for layer in self.modules[1:]:
            
            if layer.__class__ in modifier.follow:
                f_modifier = self.config.follow[layer.__class__]
                n_layer = f_modifier.identity(layer)
                evolved.append(n_layer)

            elif layer.__class__ in modifier.copy:
                if isinstance(layer, ActiSwitch): # special case
                    evolved.append(ActiSwitch(layer.activation, self.config.linear_start))
                else:
                    evolved.append(layer)

        return EvolveBlock(evolved, self.config)
    
    def summary(self, i):
        params = sum([param.numel() for param in self.modules[0].parameters()])
        print(f"Layer {i+1:<4}{self.modules[0].__class__.__name__:<25}{params:<15}", end="")
        
        activation_printed = False
        for layer in self.modules[1:]:
            if isinstance(layer, ActiSwitch):
                print(f"{layer.__class__.__name__}({layer.activation.__name__ if inspect.isfunction(layer.activation) else layer.activation.__class__.__name__}, {f'{100*(abs(layer.activation_weight.item())/(abs(layer.linear_weight.item())+abs(layer.activation_weight.item()))):.2f}':<6}%)")
                activation_printed = True
            else:
                try:
                    getattr(nn.modules.activation, layer.__class__.__name__)
                    print(f"{layer.__class__.__name__}")
                    activation_printed = True
                except:
                    pass
        if not activation_printed:
            print(f"No Activation")


class DNA:
    '''
    This class represents the DNA of a model, which is a sequence of `EvolveBlock` objects.
    It is used to store and manipulate the model's architecture during evolution.
    
    Attributes:
    -----------
    - dna (List[EvolveBlock]): A list of `EvolveBlock` instances that make up the model.
    - config (ATGENConfig): Configuration object that contains evolution parameters.
    
    Methods:
    --------
    - genes(): Returns the total number of parameters in the entire DNA sequence.
    - __str__(): Returns a string representation of the DNA sequence, including the number of genes.
    - reconstruct(): Reconstructs and returns the original `nn.Sequential` model from the DNA sequence.
    - structure(): Returns a list of the class names of the layers in the DNA sequence.
    - evolve_deeper(idx=None): Adds a new block in the DNA, increasing the model's depth.
    - evolve_wider(idx=None, remove=None): Mutates a block by increasing or decreasing its width.
    - evolve_weight(mutation_rate, perturbation_rate): Mutates the weights of the model by adding noise to some of the parameters, based on the mutation rate.
    '''
    def __init__(self, model: nn.Sequential, config: ATGENConfig):
        '''GenoType'''
        model = copy.deepcopy(model)
        self.config = config

        modules = EvolveBlock([], config)
        self.dna: List[EvolveBlock] = []
        for layer in model:
            for layer_type in config.evolve.keys():
                if isinstance(layer, layer_type):
                    if modules.full():
                        self.dna.append(modules)
                    modules = EvolveBlock([], config)
            modules.append(layer)
        
        if modules.full():
            self.dna.append(modules)

    def genes(self):
        return sum([block.genes() for block in self.dna])
    
    def __str__(self) -> str:
        return f'DNA Sequence: {len(self.dna)} evolve block of {self.genes()} genes'
    
    def reconstruct(self) -> nn.Sequential:
        '''PhenoType'''
        model = []
        for block in self.dna:
            for layer in block.modules:
                model.append(layer)
        model = nn.Sequential(*model)
        return model
    
    def structure(self):
        struct = ""
        for block in self.dna:
            struct += block.structure()
        return hash(struct)
    
    def new(self) -> nn.Sequential:
        new = []
        for block in self.dna:
            new.extend(block.new())
        model = nn.Sequential(*new)
        return model
    
    def population(self, size, device="cpu"):
        return [self.new().to(device) for _ in size]
    
    def size(self) -> List[int]:
        return [block.size() for block in self.dna]
    
    @torch.no_grad()
    def evolve_deeper(self, idx: int=None):
        if self.dna:
            idx = random.randint(0, len(self.dna)-1) if idx is None else idx
            self.dna.insert(idx, self.dna[idx].identity())
            
            if (idx == (len(self.dna)-2)) and (self.config.default_activation is not None): # add activation function if copying last layer
                if len(self.dna[-2].modules) > 1:
                    self.dna[-2].modules.pop()
                self.dna[-2].modules.append(self.config.default_activation)

    @torch.no_grad()
    def evolve_wider(self, idx: int=None, remove: bool=None):
        if remove is None:
            remove = random.random() > 0.5 if self.config.remove_mutation else False
            
        if len(self.dna) > 1: # to avoid changing output layer shape
            idx = random.randint(0, len(self.dna) - 2) if idx is None else idx
            if self.dna[idx].mutate():
                action_out = EvolveAction.DecreaseOut if remove else EvolveAction.IncreaseOut
                action_in  = EvolveAction.DecreaseIn  if remove else EvolveAction.IncreaseIn

                result = self.dna[idx].evolve(action_out)
                if isinstance(result, bool):
                    if result:
                        self.dna[idx + 1].evolve(action_in)
                else: # special case (conv->linear)
                    repeat = int(self.dna[idx + 1].modules[0].weight.shape[1]/result)
                    for _ in range(repeat):
                        self.dna[idx + 1].evolve(action_in)

    @torch.no_grad()
    def evolve_weight(self, mutation_rate, perturbation_rate):
        for param in [param for module in self.dna for layer in module for param in layer.parameters()]:
            if param.requires_grad:
                mask = torch.rand_like(param) < mutation_rate
                noise = torch.randn_like(param) * perturbation_rate  # Adjust perturbation magnitude
                param.add_(mask * noise)

    @torch.no_grad()
    def __add__(self, RNA: "DNA"):
        '''Crossover'''
        dna = copy.deepcopy(self)
        rna = copy.deepcopy(RNA)
        if dna.structure() == rna.structure():
            dna_size, rna_size = _sizes(dna.size()[:-1], rna.size()[:-1])
            for i, (t, a) in enumerate(dna_size):
                for _ in range(t):
                    dna.evolve_wider(i, a)
            for i, (t, a) in enumerate(rna_size):
                for _ in range(t):
                    rna.evolve_wider(i, a)
            dna, rna = dna.reconstruct(), rna.reconstruct()
            for param1, param2 in zip(dna.parameters(), rna.parameters()):
                mask = torch.rand_like(param1.data) > 0.5
                param1.data, param2.data = torch.where(mask, param1.data, param2.data), torch.where(mask, param2.data, param1.data)
            return DNA(dna, self.config), DNA(rna, self.config)
        else:
            raise Exception("DNA structure is not in the same Species")


    @torch.no_grad()
    def summary(self):
        print(f"{BOLD}{BLUE}Model Summary{RESET_COLOR}{BOLD}:")
        print("-" * 80)
        print(f"{'Layer':<10}{'Type':<25}{'Parameters':<15}{'Activation':<15}")
        print("-" * 80)

        for i, block in enumerate(self.dna):
            block.summary(i)

        print("-" * 80)
        print(f"{BLUE}{'Total Parameters:':<35}{RESET_COLOR}{BOLD}{self.genes():,}{RESET_COLOR}")

    
# helper crossover function
def _sizes(list1: List[int], list2: List[int]):
    offspring =  [random.randint(min(a, b), max(a, b)) for a, b in zip(list1, list2)]
    parent1 = [(o-i, True) if o>i else (i-o, False) for o, i in zip(offspring, list1)]
    parent2 = [(o-i, True) if o>i else (i-o, False) for o, i in zip(offspring, list2)]
    return parent1, parent2

if __name__ == "__main__":
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, 1, 1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 32, 3, 1, 1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.LazyLinear(1),
        nn.Softmax(-1)
    )
    model(torch.rand(64, 3, 32, 32))

    dna = DNA(model, ATGENConfig())
    dna.summary()
    # dna.evolve_deeper()
    print(model)
    print(dna)
    dna.evolve_wider()
    # for block in dna.dna:
    #     print(block.modules)
    print(dna)
    model = dna.reconstruct()
    print(model)
    print(model[0].weight.shape)
    model(torch.rand(64, 3, 32, 32))


    model = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),  # 28x28 -> 14x14
        nn.ReLU(True),
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1), # 14x14 -> 7x7
        nn.ReLU(True),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1), # 7x7 -> 4x4
        nn.ReLU(True),
        nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1), # 4x4 -> 7x7
        nn.ReLU(True),
        nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1), # 7x7 -> 14x14
        nn.ReLU(True),
        nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 14x14 -> 28x28
        nn.Sigmoid()
    )

    data_in = torch.rand(64, 1, 28, 28)
    data_out = model(data_in)
    print(data_out.shape)

    dna = DNA(model, ATGENConfig())
    dna.evolve_deeper()
    dna.evolve_wider()
    dna.evolve_deeper()
    dna.evolve_wider()

    model = dna.reconstruct()
    dna.summary()
    print(dna.structure())
    print(model)
    data_out = model(data_in)
    print(data_out.shape)

    model2 = dna.new()
    model3 = dna.new()
    print(model2[-2].parameters())
    print(model3[-2].parameters())


    model1 = nn.Sequential(
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 3)
    )
    model2 = nn.Sequential(
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 3)
    )
    model1 = DNA(model1, ATGENConfig())
    model2 = DNA(model2, ATGENConfig())
    print(model2.dna[0].size())
    print(model2.size())
    model1, model2 = (model1 + model2)
