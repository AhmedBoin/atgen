# template version

import torch
from torch import nn

from atgen.ga import ATGEN
from atgen.config import ATGENConfig
from atgen.layers.activations import ActiSwitch


class NeuroEvolution(ATGEN):
    def __init__(self, population_size: int, model: nn.Sequential, config: ATGENConfig, device):
        super().__init__(population_size, model, config, device)
        # define your own attributes and methods

    # IMPLEMENTATION REQUIRED FOR FITNESS FUNCTION..!
    def fitness_fn(self, model: nn.Sequential) -> float:
        # define your own fitness function and return fitness
        pass
    

    # no need for implementation for both, but you can implement one of them or both if you really need it
    @torch.no_grad()
    def experiences_fn(self, model: nn.Sequential):
        # define your own experiences function, no return required
        pass
    
    def backprob_fn(self, model: nn.Sequential):
        # define your own backpropagation function, no return required
        pass

    # also you can use pre_generation or post_generation if requires


# define your model or load one, it should be Sequential model
in_features: int
out_features: int
model = nn.Sequential(
    nn.Linear(in_features, out_features), 
    # it's okay if no activation required in last layer
)

# define your hyper parameters from ConfigClass
config = ATGENConfig()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
population_size: int
ne = NeuroEvolution(population_size, model, config)

# train for generation number or fitness level
ne.evolve(fitness=280, log_name="template", metrics=0, plot=True)

# load best solution
model = ne.best_individual
    
