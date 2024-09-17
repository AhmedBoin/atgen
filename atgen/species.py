import copy
import random
from typing import Dict, Iterator, List, Tuple
import numpy
from torch import nn
from tqdm import tqdm

from .config import ATGENConfig
from .dna import DNA

from .utils import BLUE, RESET_COLOR, log_level


class Individual:
    def __init__(self, dna: DNA, fitness=float("-inf"), shared=float("-inf")):
        self.dna = dna
        self.model = self.dna.reconstruct()
        self.fitness = fitness
        self.shared = shared
        self.id = self.dna.structure()


class Species:
    def __init__(self, population_size: int, dna: DNA, config: ATGENConfig):
        self.population: List[Individual] = [Individual(dna.new()) for _ in tqdm(range(population_size), desc=f"{BLUE}Initializing Population{RESET_COLOR}", ncols=100)]
        self.groups: Dict[int, int] = {}
        self.config = config
        self.calculate_species()

    def calculate_species(self):
        self.groups = {}
        for individual in self.population:
            if individual.id not in self.groups:
                self.groups[individual.id] = 1
            else:
                self.groups[individual.id] += 1

    def sort_fitness(self):
        self.population.sort(key=lambda x: x.fitness, reverse=True)

    def calculate_shared(self):
        fitness_values = numpy.exp([individual.fitness for individual in self.population])
        normalized_fitness = fitness_values/sum(fitness_values)
        for individual, shared in zip(self.population, normalized_fitness):
            individual.shared = shared / log_level(self.groups[individual.id], self.config.log_level)

    def sort_shared(self):
        self.population.sort(key=lambda x: x.shared, reverse=True)

    def genome_group(self, id) -> Tuple[List[DNA], List[float]]:
        dna = []
        fitness = []
        for individual in self.population:
            if individual.id == id:
                dna.append(individual.dna)
                fitness.append(individual.shared)
        return dna, fitness

    def select_parents(self) -> Tuple[DNA, DNA]:
        while len(self.population) == len(self.groups):
            print("TO BE HANDLE, select_parents")
            pass ## TODO:Random Evolve
        while True:
            id = random.choices(list(self.groups.keys()), list(self.groups.values()), k=1)[0]
            if self.groups[id] > 1:
                parents = self.genome_group(id)[0][:2] if self.config.select_top_only else random.choices(*self.genome_group(id), k=2)
                return parents[0], parents[1]
            
    def append(self, dna: DNA):
        self.population.append(Individual(dna))

    def extend(self, rna: List[DNA]):
        self.population.extend([Individual(dna) for dna in rna])

    def clear(self):
        self.population.clear()
        self.groups.clear()

    def __len__(self) -> int:
        return len(self.population)
    
    def __iter__(self) -> Iterator[Individual]:
        return iter(self.population)
    
    def fitness_values(self) -> Tuple[float, float, float]:
        fitness = [(individual.fitness if self.config.shared_fitness else individual.fitness) for individual in self.population]
        return max(fitness), numpy.mean(fitness), min(fitness)
    
    def best_individual(self) -> nn.Sequential:
        return self.population[0].model
    
    def best_individual_summary(self):
        self.population[0].dna.summary()

    def __getitem__(self, index):
        if isinstance(index, slice):
            new_species = copy.deepcopy(self)
            new_species.population = self.population[index]
            new_species.config = self.config
            new_species.calculate_species()
            return new_species
        elif isinstance(index, int):
            return self.population[index]
        else:
            raise TypeError(f"Invalid argument type: {type(index)}")

    

