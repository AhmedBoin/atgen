


import torch
from torch import nn

from typing import Dict, List

from ga import ATGEN

# from atgen.dna import DNA


class Species:
    def __init__(self, population: List[nn.Module], groups: Dict[int, List[nn.Module]]):
        self.population = population
        self.groups = groups











if __name__ == "__main__":
    ga = ATGEN(100, nn.Sequential(nn.Linear(8, 4)))
    species = Species(ga.population, ga.species)

    for individual in species.groups.values():
        print(len(individual))
    del species.population[:10]
    for individual in species.groups.values():
        print(len(individual))

    my_dict = {0: "A", 2: "B"}
    del my_dict[0]
    print(my_dict)