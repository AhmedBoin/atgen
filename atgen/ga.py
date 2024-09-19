import math
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


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



class ATGEN:
    def __init__(self, population_size: int, network: nn.Sequential, config=ATGENConfig()):
        self.population_size = population_size
        self.config = config
        self.metrics = 0
        
        # Initialize the population
        dna: DNA = DNA(network, config)
        self.population: Species = Species(population_size, dna, config)

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

        for individual in tqdm(self.population, desc=f"{BLUE}Fitness Evaluation{RESET_COLOR}", ncols=100):
            individual.fitness = self.fitness_fn(individual.model)

        self.population.sort_fitness()
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
        if fitness[self.metrics] > self.best_fitness: 
            self.best_fitness = fitness[self.metrics]

        if self.config.verbose:
            print_stats_table(self.best_fitness, self.metrics, fitness, len(self.population), len(self.population.groups), self.config)

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
        if self.config.save_every_generation and self.save_name is not None:
            self.save_population(self.save_name)
            self.config.save()
        
        # return if criteria reached
        if self.check_criteria(results):
            return results
        
        # implement pre_generation if required
        if self.is_overridden("pre_generation"):
            self.pre_generation()
        
        # Sort and Select top percent
        crossover_size = math.ceil((1-self.config.crossover_rate)*len(self.population)) if self.config.dynamic_dropout_population else len(self.population)//2
        self.population = self.population[:crossover_size]
        
        # offsprings generation
        offsprings: List[DNA] = []
        repeat = max(self.population_size-crossover_size, 0)
        total_mutation = repeat + len(self.population) if self.config.parent_mutation else 0
        with tqdm(total=total_mutation, desc=f"{BLUE}Crossover & Mutation{RESET_COLOR}", ncols=100) as pbar:
            if self.config.single_offspring:
                for _ in range(repeat):
                    parent1, parent2 = self.select_parents()
                    offspring = self.crossover(parent1, parent2)[0]
                    self.mutate(offspring)
                    offsprings.append(offspring)
                    pbar.update(1)
            else:
                for i in range(math.ceil(repeat/2)):
                    parent1, parent2 = self.select_parents()
                    offspring1, offspring2 = self.crossover(parent1, parent2)
                    self.mutate(offspring1)
                    offsprings.append(offspring1)
                    pbar.update(1)
                    if not ((i == math.ceil(repeat/2)-1) and ((repeat%2) == 1)):
                        self.mutate(offspring2)
                        offsprings.append(offspring2)
                        pbar.update(1)

            # mutate parents
            if self.config.parent_mutation:
                parents: List[DNA] = []
                for individual in self.population:
                    dna = individual.dna
                    self.mutate(dna)
                    parents.append(dna)
                    pbar.update(1)
                self.population.clear()
                self.population.extend(parents)
            self.population.extend(offsprings)
            self.population.calculate_species()

        # implement post_generation if required
        if self.is_overridden("post_generation"):
            self.post_generation()

        # use best genome to create experiences
        if self.is_overridden("experiences_fn"):
            self.experiences_fn(self.population.best_individual())

        # smooth the generation networks a little back propagation based method
        if self.is_overridden("backprob_fn"):
            self.evaluate_learn()

        # config modification
        self.config.step()

        return results
    
    def check_criteria(self, results): # saving criteria for reached results
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
    
    def pre_generation(self):
        raise NotImplementedError("implement pre_generation method")

    def post_generation(self):
        raise NotImplementedError("implement post_generation method")

    def save_population(self, file_name="population.pkl"):
        with open(f'{file_name}', 'wb') as file:
            pickle.dump(self.population, file)

    def load_population(self, file_name="population.pkl"):
        with open(f'{file_name}', 'rb') as file:
            self.population = pickle.load(file)
    
    def continual_evolution(self, fitness):
        self.evaluate_fitness()
        fitness_scores = [individual.fitness for individual in self.population]
        minimum_fitness = min(fitness_scores)
        generation = 1
        while True:
            for i in range(self.population_size):
                parent1, parent2 = self.select_parents()
                offspring = self.crossover(parent1, parent2)[0]
                self.mutate(offspring)
                offspring = Individual(offspring)
                offspring.fitness = self.fitness_fn(offspring.model)
                if offspring.fitness > minimum_fitness:
                    self.population.exchange(offspring, minimum_fitness)
                    fitness_scores = [individual.fitness for individual in self.population]
                    minimum_fitness = min(fitness_scores)
                    self.population.calculate_shared()
                    if self.config.shared_fitness:
                        self.population.sort_shared()
                if offspring.fitness > self.best_fitness:
                    self.best_fitness = offspring.fitness
                    print(f"Generation: {generation}, Better Fitness Reached: {offspring.fitness}")
                else:
                    print(f"Generation: {generation}, Offspring Fitness: {offspring.fitness}", end="\r")
                if self.best_fitness >= fitness:
                    return
            generation += 1
            if self.is_overridden("pre_generation"):
                self.pre_generation()
            self.config.step()
            print("Mutation Rate:",self.config.mutation_rate)
            self.save_population()


    
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

