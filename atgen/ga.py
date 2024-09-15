import math
import numpy
import torch
from torch import nn

from tqdm import tqdm
import multiprocessing
import matplotlib.pyplot as plt

import inspect
import copy
import random
from typing import Dict, List, Tuple
import pickle

from .dna import DNA
from .utils import (RESET_COLOR, BLUE, GREEN, RED, BOLD, GRAY, 
                   print_stats_table, merge_dicts, shift_to_positive, log_level)
from .config import ATGENConfig


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



class ATGEN:
    def __init__(self, population_size: int, network: nn.Sequential, config=ATGENConfig(), device="cpu"):
        self.population_size = population_size
        self.config = config
        self.metrics = 0
        
        # Initialize the population
        dna = DNA(network, config)
        self.population: Dict[int, List[Tuple[nn.Sequential, float]]] = {}
        for _ in tqdm(range(population_size), desc=f"{BLUE}Initializing Population{RESET_COLOR}", ncols=100):
            species, genome = dna.new()
            if species in self.population:
                self.population[species].append([genome, 0.0])
            else:
                self.population[species] = [[genome, 0.0]]
        self.fitness_scores = []
        self.shared_fitness = []
        self.best_individual = network 

        # Preview results
        self.best_fitness = float("-inf")
        self.last_fitness = float("-inf")
        self.worst_shared = float("-inf")
        self.required_fitness: float = None

    @torch.no_grad()
    def evaluate_fitness(self):
        """
        Evaluate the fitness of each network in the 
        """

        self.fitness_scores = []
        self.shared_fitness = []
        current_population = sum([len(species) for species in self.population.values()])

        with tqdm(total=current_population, desc=f"{BLUE}Fitness Evaluation{RESET_COLOR}", ncols=100) as pbar:
            for species in self.population.values():
                for i in range(len(species)):
                    species[i][1] = self.fitness_fn(species[i][0])
                    self.fitness_scores.append(species[i][1])
                    if species[i][1] > self.best_fitness:
                        self.best_individual = species[i][0]
                    pbar.update(1)
        
        if self.config.shared_fitness:
            self.worst_shared = min(self.fitness_scores)-1
            self.shared_fitness = [fitness-self.worst_shared for fitness in self.fitness_scores]
            counter = 0
            for species in self.population.values():
                for _ in range(len(species)):
                    self.shared_fitness[counter] = self.shared_fitness[counter]/log_level(len(species), self.config.log_level)
                    counter += 1

    
    def evaluate_learn(self):
        '''backpropagation phase'''
        current_population = sum([len(species) for species in self.population.values()])

        with tqdm(total=current_population, desc=f"{BLUE}Generation Refinement{RESET_COLOR}", ncols=100) as pbar:
            for species in self.population.values():
                for i in range(len(species)):
                    self.backprob_fn(species[i][0])
                    pbar.update(1)

    def select_parents(self) -> Tuple[DNA, DNA]:
        """
        Select two parents from the population based on fitness scores using roulette wheel selection.
        
        Returns:
            Tuple[DNA, DNA]: Two selected parent networks.
        """
        while True:
            species: int = random.choices(list(self.population.keys()), weights=[len(species) for species in self.population.values()], k=1)[0]
            if len(self.population[species]) > 1:
                break
            
        weights = shift_to_positive([fitness[1] for fitness in self.population[species]])
        parents = random.choices(self.population[species], weights=weights, k=2)
        parent1, parent2 = DNA(parents[0][0], self.config), DNA(parents[1][0], self.config)
        
        return parent1, parent2

    @torch.no_grad()
    def crossover(self, parent1: DNA, parent2: DNA) -> Tuple[DNA, DNA]:
        """
        Perform a more vectorized crossover between two parent networks to produce a offspring network.
        
        Args:
            parent1 (DNA): The first parent network.
            parent2 (DNA): The second parent network.
        
        Returns:
            DNA: A new offspring network created from crossover.
        """
        
        offspring1, offspring2 = parent1 + parent2

        return offspring1, offspring2

    @torch.no_grad()
    def mutate(self, network: DNA):
        """
        Mutate the given network with a small probability.
        
        Args:
            network (DNA): The network to mutate.
        """
        if random.random() < self.config.deeper_mutation:
            network.evolve_deeper()
        if random.random() < self.config.wider_mutation:
            network.evolve_wider()
        network.evolve_weight(self.config.mutation_rate, self.config.perturbation_rate)

    def filter_population(self, fitness_value):
        for species in self.population.values():
            length = log_level(len(species), self.config.log_level)
            for i in reversed(range(len(species))):
                if self.config.shared_fitness:
                    if ((species[i][1]-self.worst_shared)/length) <= fitness_value:
                        del species[i]
                else:
                    if species[i][1] <= fitness_value:
                        del species[i]

    def preview_results(self):
        fitness = [max(self.fitness_scores), numpy.mean(self.fitness_scores), min(self.fitness_scores)]
        color = GREEN if fitness[self.metrics] > self.best_fitness else GRAY if fitness[self.metrics] > self.last_fitness else RED

        print(f"{BLUE}Best Fitness{RESET_COLOR}: \t {BOLD}{color}{fitness[self.metrics]}{RESET_COLOR}")
        print(f"{BLUE}{BOLD}Best{RESET_COLOR}", end=" ")
        DNA(self.best_individual, self.config).summary()

        self.last_fitness = fitness[self.metrics] 
        if fitness[self.metrics] > self.best_fitness: 
            self.best_fitness = fitness[self.metrics]

        if self.config.verbose:
            current_population = sum([len(species) for species in self.population.values()])
            print_stats_table(self.best_fitness, self.metrics, fitness, current_population, len(self.population), self.config)

        # Debug for fitness ##################################################################################
        # for key, species in self.population.items():
            # print(len(species), max([item[1] for item in species]))
            # print(f"size: {len(species):<3}, {max([item[1] for item in species]):<7}, Id: {key}")

        return fitness
                
    def run_generation(self) -> List[float]:
        """
        Run a single generation of the genetic algorithm, including evaluation, selection, crossover, mutation, and pruning.
        
        Args:
            fitness_fn (function): A function to evaluate the fitness of a network.
        """
        
        # Evaluate the fitness and preview results
        self.evaluate_fitness()
        results = self.preview_results()
        
        # return if criteria reached
        if self.check_criteria(results):
            return results
        
        # Sort and Select top percent
        current_population = sum([len(species) for species in self.population.values()])
        crossover_size = math.ceil((1-self.config.crossover_rate)*current_population) if self.config.dynamic_dropout_population else current_population//2
        fitness_score = sorted(self.shared_fitness if self.config.shared_fitness else self.fitness_scores, reverse=True)
        self.filter_population(fitness_score[crossover_size])
        
        # offsprings generation
        offsprings: Dict[int, List[Tuple[nn.Sequential, float]]] = {}
        repeat = max(self.population_size-crossover_size, 0)
        total_mutation = repeat + sum([len(species) for species in self.population.values()]) if self.config.parent_mutation else 0
        with tqdm(total=total_mutation, desc=f"{BLUE}Crossover & Mutation{RESET_COLOR}", ncols=100) as pbar:
            if self.config.single_offspring:
                for _ in range(repeat):
                    parent1, parent2 = self.select_parents()
                    offspring = self.crossover(parent1, parent2)[0]
                    self.mutate(offspring)
                    key = offspring.structure()
                    offspring = offspring.reconstruct()
                    if key in offsprings:
                        offsprings[key].append([offspring, float("-inf")])
                    else:
                        offsprings[key] = [[offspring, float("-inf")]]
                    pbar.update(1)
            else:
                for i in range(math.ceil(repeat/2)):
                    parent1, parent2 = self.select_parents()
                    offspring1, offspring2 = self.crossover(parent1, parent2)
                    self.mutate(offspring1)
                    key1 = offspring1.structure()
                    offspring1 = offspring1.reconstruct()
                    if key1 in offsprings:
                        offsprings[key1].append([offspring1, float("-inf")])
                    else:
                        offsprings[key1] = [[offspring1, float("-inf")]]
                    pbar.update(1)
                    if not ((i == math.ceil(repeat/2)-1) and ((repeat%2) == 1)):
                        self.mutate(offspring2)
                        key2 = offspring2.structure()
                        offspring2 = offspring2.reconstruct()
                        if key2 in offsprings:
                            offsprings[key2].append([offspring2, float("-inf")])
                        else:
                            offsprings[key2] = [[offspring2, float("-inf")]]
                        pbar.update(1)
                # if (repeat%2) == 1:
                #     offsprings[key2].pop()
                #     if len(offsprings[key2]) == 0:
                #         del offsprings[key2]

            # mutate parents
            if self.config.parent_mutation:
                parents: Dict[int, List[Tuple[nn.Sequential, float]]] = {}
                for species in self.population.values():
                    for individual in species:
                        parent = DNA(individual[0], self.config)
                        self.mutate(parent)
                        key = parent.structure()
                        parent = parent.reconstruct()
                        if key in parents:
                            parents[key].append([parent, float("-inf")])
                        else:
                            parents[key] = [[parent, float("-inf")]]
                        pbar.update(1)
                self.population = merge_dicts(parents, offsprings)
            else:
                self.population = merge_dicts(self.population, offsprings)

        # use best genome to create experiences
        if self.is_overridden("experiences_fn"):
            self.experiences_fn(self.best_individual)

        # smooth the generation networks a little back propagation based method
        if self.is_overridden("backprob_fn"):
            self.evaluate_learn()

        # config modification
        self.config.crossover_step()
        self.config.mutation_step()
        self.config.perturbation_step()
        if self.config.save_every_generation and self.save_name is not None:
            self.save_population(self.save_name)
            self.config.save()

        return results
    
    def check_criteria(self, results):
        return results[self.metrics] > self.required_fitness if self.required_fitness else False

    def evolve(self, generation: int=None, fitness: int=None, save_name: str=None, metrics: int=0, plot: bool=False):
        self.metrics = metrics
        self.save_name = save_name
        self.required_fitness = fitness
        last_fitness = -float("inf")
        maximum, mean, minimum = [], [], []
        if generation is not None:
                for i in range(generation):
                    print(f"\n--- {BLUE}Generation {i+1}{RESET_COLOR} ---")
                    results = self.run_generation()
                    maximum.append(results[0])
                    mean.append(results[1])
                    minimum.append(results[2])
                    last_fitness = results[metrics]
                    if fitness is not None:
                        if last_fitness >= fitness:
                            print(f"\n--- {BLUE}Fitness reached{RESET_COLOR} ---")
                            break
        
        elif fitness is not None:
            generation =  0
            while fitness > last_fitness:
                generation += 1
                print(f"\n--- {BLUE}Generation {generation}{RESET_COLOR} ---")
                results = self.run_generation()
                maximum.append(results[0])
                mean.append(results[1])
                minimum.append(results[2])
                last_fitness = results[metrics]
            print(f"\n--- {BLUE}Fitness reached{RESET_COLOR} ---")
            
        else:
            raise ValueError("evolve using generation number or fitness value")
        
        if save_name is not None:
            self.save_population(save_name)
        
        if plot:
            plt.plot(range(len(mean)), mean, label='Mean Fitness')
            plt.fill_between(range(len(mean)), minimum, maximum, alpha=0.2, label='Confidence Interval')

            # Set labels and title
            plt.xlabel('Generation Number')
            plt.ylabel('Fitness')
            plt.title('Training Progress with Confidence Interval')
            plt.legend(loc='upper right')
            plt.grid(True)
            plt.show()
        else:
            return maximum, mean, minimum
    
    def fitness_fn(self, genome):
        raise NotImplementedError("implement fitness_fn method")
    
    def backprob_fn(self, genome):
        raise NotImplementedError("implement learn_fn method")
    
    def experiences_fn(self, genome):
        raise NotImplementedError("implement store_experiences method")
    
    def is_overridden(self, method_name):
        if getattr(self, method_name, None) is None:
            return False
        
        for cls in inspect.getmro(self.__class__):
            if method_name in cls.__dict__:
                return cls != ATGEN
        return False

    def save_population(self, file_name="population.pkl"):
        with open(f'{file_name}', 'wb') as file:
            pickle.dump(self.population, file)

    def load_population(self, file_name="population.pkl"):
        with open(f'{file_name}', 'rb') as file:
            self.population = pickle.load(file)
    

def evaluate_fitness_single(args):
    """
    Helper function to evaluate the fitness of a single network.
    
    Args:
        args (tuple): A tuple containing the index of the network and the network object itself.
    
    Returns:
        tuple: A tuple containing the index and the fitness score.
    """
    index, network, fitness_fn = args
    fitness_score = fitness_fn(network)
    return (index, fitness_score)

def refine_genome(args):
    genome, learn_fn = args
    learn_fn(genome)
    return genome 


    
if __name__ == "__main__":
    # Initialize the ATGEN instance
    model = nn.Sequential(
        nn.Linear(8, 4),
        nn.ReLU(),
        nn.Linear(4, 1)
    )
    ga = ATGEN(population_size=10, network=model)
    
    # Create parent networks with specified architectures
    parent1 = DNA(ga.population[0][0], ATGENConfig())
    parent2 = DNA(ga.population[0][1], ATGENConfig())
    
    # Perform crossover
    offspring1 = ga.crossover(parent1, parent2)[0]
    offspring1.summary()
    
    # # Create CNN parent networks
    # parent1 = nn.Sequential(
    #     Conv2D(3, 32),
    #     ActiSwitch(nn.ReLU()),
    #     MaxPool2D(),
    #     Conv2D(32, 64),
    #     ActiSwitch(nn.ReLU()),
    #     MaxPool2D(),
    #     Flatten(),
    #     LazyLinear(100),
    #     ActiSwitch(nn.ReLU()),
    #     Linear(100, 10),
    #     input_size=(32, 32)
    # )
    # parent2 = nn.Sequential(
    #     Conv2D(3, 32),
    #     ActiSwitch(nn.ReLU()),
    #     MaxPool2D(),
    #     Conv2D(32, 64),
    #     ActiSwitch(nn.ReLU()),
    #     Conv2D(64, 64),
    #     ActiSwitch(nn.ReLU()),
    #     MaxPool2D(),
    #     Flatten(),
    #     LazyLinear(200),
    #     ActiSwitch(nn.ReLU()),
    #     Linear(200, 100),
    #     ActiSwitch(nn.ReLU()),
    #     Linear(100, 10),
    #     input_size=(32, 32)
    # )
    # parent1.summary()
    # parent2.summary()
    # # summary(parent1, (3, 32, 32), device="cpu")
    # # summary(parent2, (3, 32, 32), device="cpu")
    
    # # Perform crossover
    # parent2 = parent2.genotype()
    # print(len(parent2.conv))
    # parent2.evolve_conv_layer(1)
    # offspring = ga.crossover(parent1.genotype(), parent2)
    # offspring = ATNetwork.phenotype(offspring)
    # offspring.summary()
    # # summary(offspring, (3, 32, 32), device="cpu")

