import copy
import math
import os
import torch
from torch import nn

from tqdm import tqdm
import multiprocessing
import concurrent.futures
import matplotlib.pyplot as plt

import inspect
import random
from typing import Dict, List, Tuple
import pickle


from .species import Individual, Species
from .dna import DNA
from .config import ATGENConfig
from .utils import RESET_COLOR, BLUE, GREEN, RED, BOLD, GRAY, print_stats_table
from .memory import Action, ReplayBuffer


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



class ATGEN:
    def __init__(self, population_size: int, network: nn.Sequential, config=ATGENConfig(), memory=ReplayBuffer(), patience=None):
        self.population_size = population_size
        self.config = config
        self.memory = memory
        self.metrics = 0
        
        # Initialize the population
        dna: DNA = DNA(network, config)
        self.population: Species = Species(population_size, dna, config)
        self.best_individual = self.population.population[0]

        # Preview results
        self.best_fitness = float("-inf")
        self.last_fitness = float("-inf")
        self.worst_shared = float("-inf")
        self.last_improvement = 0 # to track patient flags in Memory and Config classes

        self.extra = None # save utils (custom objects) used in your design

        # stopping criteria
        self.required_fitness: float = None
        self.fitness_reached = False # avoid double criteria check (avoid double difficulty)
        self.patience = patience

    @torch.no_grad()
    def evaluate_fitness(self):
        """
        Evaluate the fitness of each network in the 
        """

        for individual in tqdm(self.population, desc=f"{BLUE}Fitness Evaluation{RESET_COLOR}", ncols=100):
            total_fitness = 0
            for _ in range(self.config.current_difficulty):
                total_fitness += self.fitness_fn(individual.model)
            individual.fitness = total_fitness / self.config.current_difficulty

        self.population.sort_fitness()
        if self.population.population[0].fitness > self.best_fitness:
            self.best_individual = self.population.population[0]
        self.population.calculate_shared()
        if self.config.shared_fitness:
            self.population.sort_shared()

    
    def evaluate_learn(self):
        '''backpropagation phase'''

        # for individual in tqdm(self.population, desc=f"{BLUE}Generation Refinement{RESET_COLOR}", ncols=100):
        #     self.backprob_fn(individual.model)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            list(tqdm(executor.map(lambda individual: self.backprob_fn(individual.model), self.population), desc=f"{BLUE}Generation Refinement{RESET_COLOR}", ncols=100, total=len(self.population)))


    def select_parents(self) -> Tuple[DNA, DNA]:
        """
        Select two parents from the population based on fitness scores using roulette wheel selection.
        
        Returns:
            Tuple[DNA, DNA]: Two selected parent networks.
        """
        
        return self.population.select_parents()

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

        return parent1 + parent2

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

    def preview_results(self):
        fitness = self.population.fitness_values()
        color = GREEN if fitness[self.metrics] > self.best_fitness else GRAY if fitness[self.metrics] > self.last_fitness else RED

        print(f"{BLUE}Best Fitness{RESET_COLOR}: \t {BOLD}{color}{fitness[self.metrics]}{RESET_COLOR}")
        print(f"{BLUE}{BOLD}Best{RESET_COLOR}", end=" ")
        self.population.best_individual_summary()

        self.last_fitness = fitness[self.metrics] 
        if self.last_fitness > self.best_fitness: 
            self.best_fitness = self.last_fitness
            self.last_improvement = 0
        else:
            self.last_improvement += 1
            if self.config.patience == self.last_improvement:
                if self.config.current_depth < self.config.maximum_depth:
                    self.config.current_depth += 1
            if self.memory.patience == self.last_improvement:
                self.memory.clear()

        if self.config.verbose:
            print_stats_table(self.best_fitness, self.metrics, fitness, len(self.population), len(self.population.groups), self.config)
            if self.memory.is_available():
                print(f"{BLUE}Experience Buffer{RESET_COLOR}: {len(self.memory)}\
                \n -> {GREEN}Good{RESET_COLOR} (Size: {len(self.memory.good_buffer):<2}, Upper Bound: {round(self.memory.upper_bound, 3):<6}, Minimum Reward: {round(self.memory.good_buffer.min, 3):<6}) \
                \n -> {RED}Bad {RESET_COLOR} (Size: {len(self.memory.bad_buffer):<2}, Lower Bound: {round(self.memory.lower_bound, 3):<6}, Maximum Reward: {round(self.memory.bad_buffer.max, 3):<6})")
                    
        return fitness
                
    def run_generation(self) -> List[float]:
        """
        Run a single generation of the genetic algorithm, including evaluation, selection, crossover, mutation, and pruning.
        
        Args:
            fitness_fn (function): A function to evaluate the fitness of a network.
        """
        
        # Evaluate the fitness and preview results, also save it if required
        self.evaluate_fitness()
        results = self.preview_results()
        if self.config.save_every_generation and self.log_name is not None:
            self.save(self.log_name, self.extra)
        
        # return if criteria reached
        if self.check_criteria(results):
            return results


        # implement pre_generation if required
        if self.is_overridden("pre_generation"):
            print(f"{BLUE}Pre-Generation Fn in Execution...{RESET_COLOR}")
            self.pre_generation()
        
        # Sort and Select top percent
        crossover_size = math.ceil((1-self.config.crossover_rate)*len(self.population)) if self.config.dynamic_dropout_population else len(self.population)//2
        self.population = self.population[:crossover_size]
        
        # offsprings generation
        offsprings: List[DNA] = []
        repeat = max(self.population_size-crossover_size, 0)
        total_mutation = repeat + (0 if self.config.elitism else len(self.population))
        with tqdm(total=total_mutation, desc=f"{BLUE}Crossover & Mutation{RESET_COLOR}", ncols=100) as pbar:
            if self.config.single_offspring:
                for _ in range(repeat):
                    offspring = None
                    while not offspring:
                        parent1, parent2 = self.select_parents()
                        offspring = self.crossover(parent1, parent2)[0]
                        self.mutate(offspring)
                        offspring = self.memory.validate(offspring.reconstruct())
                    offsprings.append(DNA(offspring, self.config))
                    pbar.update(1)
            else:
                for i in range(math.ceil(repeat/2)):
                    parent1, parent2 = self.select_parents()
                    offspring1, offspring2 = self.crossover(parent1, parent2)
                    self.mutate(offspring1)
                    self.mutate(offspring2)
                    offsprings.append(offspring1)
                    pbar.update(1)
                    if not ((i == math.ceil(repeat/2)-1) and ((repeat%2) == 1)):
                        offsprings.append(offspring2)
                        pbar.update(1)

            # mutate parents
            if not self.config.elitism:
                parents: List[DNA] = []
                for individual in self.population:
                    dna = None
                    while not dna:
                        dna = copy.deepcopy(individual.dna)
                        self.mutate(dna)
                        dna = self.memory.validate(dna.reconstruct())
                    parents.append(DNA(dna, self.config))
                    pbar.update(1)
                self.population.clear()
                self.population.extend(parents)
            self.population.extend(offsprings)
            self.population.calculate_species()

        # implement post_generation if required
        if self.is_overridden("post_generation"):
            print(f"{BLUE}Post-Generation Fn in Execution...{RESET_COLOR}")
            self.post_generation()

        # use best genome to create experiences
        if self.is_overridden("experiences_fn"):
            print(f"{BLUE}Experience Fn phase in Execution.{RESET_COLOR}")
            self.experiences_fn(self.population.best_individual())

        # smooth the generation networks a little back propagation based method
        if self.is_overridden("backprob_fn"):
            print(f"{BLUE}Back-Propagation phase in Execution.{RESET_COLOR}")
            self.evaluate_learn()

        # config modification
        self.config.step()

        return results
    
    
    def check_criteria(self, results): # saving criteria for reached results
        if self.required_fitness:
            if results[self.metrics] >= self.required_fitness:
                if self.config.current_difficulty < self.config.difficulty:
                    self.config.current_difficulty += 1
                    self.memory.clear()
                    self.best_fitness = float("-inf")
                    self.last_fitness = float("-inf")
                    self.worst_shared = float("-inf")
                elif self.config.current_difficulty == self.config.difficulty:
                    self.fitness_reached = True
                    print(f"--- {BLUE}Fitness reached{RESET_COLOR} ---\n")
                    return True
        if self.patience:
            if self.patience == self.last_improvement:
                self.fitness_reached = True
                print(f"--- {BLUE}Fitness reached{RESET_COLOR} ---\n")
                return True
        return False
    

    def evolve(self, generation: int=None, fitness: int=None, log_name: str=None, metrics: int=0, plot: bool=False):
        self.metrics = metrics
        self.log_name = log_name
        self.required_fitness = fitness

        maximum, mean, minimum = [], [], []
        if generation is not None:
                for i in range(generation):
                    print(f"\n--- {BLUE}Generation {i+1}{RESET_COLOR} ---")
                    results = self.run_generation()
                    maximum.append(results[0])
                    mean.append(results[1])
                    minimum.append(results[2])
                    if self.fitness_reached:
                        break
        # reach fitness if no generation number required
        elif fitness is not None:
            generation =  0
            while True:
                generation += 1
                print(f"\n--- {BLUE}Generation {generation}{RESET_COLOR} ---")
                results = self.run_generation()
                maximum.append(results[0])
                mean.append(results[1])
                minimum.append(results[2])
                if self.fitness_reached:
                    break
        # break if no given generation number or fitness value
        else:
            raise ValueError("evolve using generation number or fitness value")
        
        if plot:
            plt.plot(range(len(mean)), mean, label='Mean Fitness')
            plt.fill_between(range(len(mean)), minimum, maximum, alpha=0.2, label='Confidence Interval')
            plt.xlabel('Generation Number')
            plt.ylabel('Fitness')
            plt.title('Training Progress with Confidence Interval')
            plt.legend(loc='upper right')
            plt.grid(True)
            plt.show()
        return maximum, mean, minimum
    
    
    def is_overridden(self, method_name):
        if getattr(self, method_name, None) is None:
            return False
        
        for cls in inspect.getmro(self.__class__):
            if method_name in cls.__dict__:
                return cls != ATGEN
        return False
    
    def fitness_fn(self, genome):
        raise NotImplementedError("implement fitness_fn method")
    
    def backprob_fn(self, genome):
        raise NotImplementedError("implement learn_fn method")
    
    def experiences_fn(self, genome):
        raise NotImplementedError("implement store_experiences method")
    
    def pre_generation(self):
        raise NotImplementedError("implement pre_generation method")

    def post_generation(self):
        raise NotImplementedError("implement post_generation method")

    def save_population(self, file_name):
        with open(f'{file_name}', 'wb') as file:
            pickle.dump(self.population, file)

    def load_population(self, file_name):
        with open(f'{file_name}', 'rb') as file:
            self.population = pickle.load(file)

    def save_individual(self, file_name):
        with open(f'{file_name}', 'wb') as file:
            pickle.dump(self.best_individual, file)

    def load_individual(self, file_name):
        with open(f'{file_name}', 'rb') as file:
            self.best_individual = pickle.load(file)

    def save(self, directory="model", extra=None):
        os.makedirs(f"logs/{directory}", exist_ok=True)
        self.save_population(f"logs/{directory}/population.pkl")
        self.save_individual(f"logs/{directory}/individual.pkl")
        self.config.save(f"logs/{directory}/config.json")
        self.memory.save(f"logs/{directory}/memory.pkl")
        if extra:
            with open(f"logs/{directory}/extra.pkl", "wb") as file:
                pickle.dump(extra, file)

    def load(self, directory="model"):
        extra = None
        if os.path.exists(f"logs/{directory}"):
            self.load_population(f"logs/{directory}/population.pkl")
            self.load_individual(f"logs/{directory}/individual.pkl")
            self.config = self.config.load(f"logs/{directory}/config.json")
            self.memory = self.memory.load(f"logs/{directory}/memory.pkl")
            if os.path.exists(f"logs/{directory}/extra.pkl"):
                with open(f"logs/{directory}/extra.pkl", "rb") as file:
                    extra = pickle.load(file)
        else:
            warnings.warn(f"Log path 'logs/{directory}' not found. Starting a new training session.", UserWarning)
        return extra


    
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

