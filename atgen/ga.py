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

from layers import ActiSwitch
from dna import DNA
from utils import RESET_COLOR, BLUE, GREEN, RED, BOLD, GRAY, print_stats_table
from config import ATGENConfig


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



class ATGEN(nn.Module):
    def __init__(self, population_size: int, network: nn.Sequential, config=ATGENConfig(), device="cpu"):
        """
        Initialize the Genetic Algorithm for evolving neural networks.

        Args:
            population_size (int): The number of networks in the population.
            layers (List[int]): The structure of layers for initial networks.
            activation (function): Activation function for hidden layers.
            last_activation (function): Activation function for the last layer.
            bias (bool): Whether to use bias in layers.
            mutation_rate (float): Probability of mutating a network.
            crossover_rate (float): Probability of performing crossover between two networks.
            threshold (float): Threshold for pruning neurons with low weights.
        """
        self.population_size = population_size
        self.config = config
        self.metrics = 0
        
        # Initialize the population
        dna = DNA(network, config)
        self.population = [dna.new().to(device) for _ in range(population_size)]
        self.species: Dict[int, List[nn.Sequential]] = {dna.structure(): self.population}
        self.fitness_scores = [0.0] * population_size
        self.group_size = [population_size] * population_size
        self.shared_fitness = copy.deepcopy(self.fitness_scores)
        self.selection_probs = copy.deepcopy(self.fitness_scores)
        self.best_fitness = float("-inf")
        self.last_fitness = float("-inf")

    @torch.no_grad()
    def evaluate_fitness(self):
        """
        Evaluate the fitness of each network in the population.
        """
        self.fitness_scores = [0.0] * len(self.population)
        self.shared_fitness = [0.0] * len(self.population)

        for i, network in enumerate(tqdm(self.population, desc=f"{BLUE}Fitness Evaluation{RESET_COLOR}", ncols=100)):
            self.fitness_scores[i] = self.fitness_fn(network)
            self.shared_fitness[i] = self.fitness_scores[i] / self.group_size[i]
        
        # args = [(i, network, self.fitness_fn) for i, network in enumerate(self.population)]
        # with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        #     for index, fitness_score in tqdm(pool.imap_unordered(evaluate_fitness_single, args), total=len(args), desc=f"{BLUE}Fitness Evaluation{RESET_COLOR}", ncols=100):
        #         self.fitness_scores[index] = fitness_score

    def evaluate_learn(self):
        """
        """
        for genome in tqdm(self.population, desc=f"{BLUE}Generation Refinement{RESET_COLOR}", ncols=100):
            self.backprob_fn(genome)

        # args = [(genome, self.learn_fn) for genome in self.population]
        # with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        #     for _ in tqdm(pool.imap_unordered(refine_genome, args), total=len(args), desc=f"{BLUE}RL Generation Refinement{RESET_COLOR}", ncols=100):
        #         pass

    def select_parents(self) -> Tuple[DNA, DNA]:
        """
        Select two parents from the population based on fitness scores using roulette wheel selection.
        
        Returns:
            Tuple[DNA, DNA]: Two selected parent networks.
        """
        parent1 = DNA(random.choices(self.population, weights=self.selection_probs, k=1)[0], self.config)
        parent2 = DNA(random.choices(self.population, weights=self.selection_probs, k=1)[0], self.config)
        
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
        offspring1, offspring2 = parent1 + parent1

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

                
    def run_generation(self) -> Tuple[float, float, float]:
        """
        Run a single generation of the genetic algorithm, including evaluation, selection, crossover, mutation, and pruning.
        
        Args:
            fitness_fn (function): A function to evaluate the fitness of a network.
        """

        # Set generation size:
        crossover_size = int((1-self.config.crossover_rate)*self.population_size) if self.config.dynamic_dropout_population else self.population_size//2
        
        # Evaluate the fitness of each network
        self.evaluate_fitness()

        # preview training data
        max_fitness = max(self.fitness_scores)
        min_fitness = min(self.fitness_scores)
        mean_fitness = numpy.mean(self.fitness_scores)
        all_fitness = [max_fitness, mean_fitness, mean_fitness]
        color = GREEN if all_fitness[self.metrics] > self.best_fitness else GRAY if all_fitness[self.metrics] > self.last_fitness else RED
        print_current_best = f"{BLUE}Best Fitness{RESET_COLOR}: \t {BOLD}{color}{ all_fitness[self.metrics] }{RESET_COLOR}"
        self.last_fitness = all_fitness[self.metrics] 
        if all_fitness[self.metrics] > self.best_fitness: 
            self.best_fitness = all_fitness[self.metrics]

        # select sorting type
        fitness_score = self.shared_fitness if self.config.shared_fitness else self.fitness_scores

        # selection of parents
        fits = sorted([fit for fit in fitness_score], reverse=True)[:crossover_size]
        min_fit = min(fits)
        fits = [fit-min_fit for fit in fits] # make all positive numbers
        total_fitness = sum(fits)
        self.selection_probs = [score / total_fitness for score in fits] # convert to probabilities
        
        sorted_population = [ind for _, ind in sorted(zip(fitness_score, self.population), key=lambda x: x[0], reverse=True)]
        self.population = sorted_population[:crossover_size]  # Select top percent
        
        # offsprings generation
        offsprings = []
        if self.config.single_offspring:
            repeat = self.population_size-crossover_size
            for _ in tqdm(range(repeat), desc=f"{BLUE}Crossover & Mutation{RESET_COLOR}", ncols=100):
                parent1, parent2 = self.select_parents()
                offspring = self.crossover(parent1, parent2)[0]
                self.mutate(offspring)
                offspring = offspring.reconstruct()
                offsprings.append(offspring)
        else:
            for _ in tqdm(range(math.ceil(repeat/2)), desc=f"{BLUE}Crossover & Mutation{RESET_COLOR}", ncols=100):
                parent1, parent2 = self.select_parents()
                offspring1, offspring2 = self.crossover(parent1, parent2)
                self.mutate(offspring1)
                self.mutate(offspring2)
                offspring1 = offspring1.reconstruct()
                offspring2 = offspring2.reconstruct()
                offsprings.append(offspring1)
                offsprings.append(offspring2)
            if (repeat%2) == 1:
                offsprings.pop()

        # mutate parents
        if self.config.parent_mutation:
            parents = []
            for individual in self.population:
                individual = DNA(individual, self.config)
                self.mutate(individual)
                parents.append(individual.reconstruct())
            self.population = parents + offsprings
        else:
            self.population.extend(offsprings)

        # species individual into subgroups
        self.species = {}
        for individual in self.population:
            individual_id = DNA(individual, self.config).structure()
            if individual in self.species:
                self.species[individual_id].append(individual)
            else:
                self.species[individual_id] = [individual]
                
        for i, individual in enumerate(self.population):
            for key in self.species.keys():
                if DNA(individual, self.config).structure() == key:
                    self.group_size[i] = len(self.species[key])
                    break

        # results
        print(print_current_best)
        print(f"{BLUE}{BOLD}Best{RESET_COLOR}", end=" ")
        DNA(self.population[0], self.config).summary()

        # use best genome to create experiences
        if self.is_overridden("experiences_fn"):
            self.experiences_fn(self.population[0])

        # smooth the generation networks a little back propagation based method
        if self.is_overridden("backprob_fn"):
            self.evaluate_learn()

        # config modification
        print_stats_table(self.best_fitness, self.metrics, max_fitness, mean_fitness, min_fitness, self.population_size, len(self.species))
        self.config.crossover_step()
        self.config.mutation_step()
        if self.config.save_every_generation and self.save_name is not None:
            self.save_population(self.save_name)

        return max_fitness, mean_fitness, min_fitness

    def evolve(self, generation: int=None, fitness: int=None, save_name: str=None, metrics: int=0, plot: bool=False):
        self.metrics = metrics
        self.save_name = save_name
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
        
        # rearrange
        self.evaluate_fitness()
        self.population = [ind for _, ind in sorted(zip(self.shared_fitness, self.population), key=lambda x: x[0], reverse=True)]
        
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
    parent1 = DNA(ga.population[0], ATGENConfig())
    parent2 = DNA(ga.population[1], ATGENConfig())
    
    # Perform crossover
    offspring1 = ga.crossover(parent1, parent2)[0]
    offspring1.summary()
    
    # # Create CNN parent networks
    # parent1 = ATNetwork(
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
    # parent2 = ATNetwork(
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

