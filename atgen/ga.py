import numpy
import torch
from torch import nn

from tqdm import tqdm
import multiprocessing
import matplotlib.pyplot as plt

import inspect
import copy
import random
from typing import List, Tuple
import pickle

from network import ATNetwork
from layers import Linear, Conv2D, LazyConv2D, MaxPool2D, Flatten, ActiSwitch, Pass, LazyLinear
from dna import DNA
from utils import RESET_COLOR, BLUE, GREEN, RED, BOLD, GRAY, activation_functions, print_stats_table


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



class ATGEN(nn.Module):
    def __init__(self, population_size: int, layers: List[int], activation=nn.ReLU(), last_activation=None, bias=True, 
                 crossover_rate = 0.8, weight_mutation_rate=0.01, perturbation_rate=0.9, add_neuron_mutation_rate=0.2, 
                 add_filter_mutation_rate=0.0, linear_mutation_rate=0.0, conv_mutation_rate=0.0, activation_mutation_rate=0.001, 
                 threshold=0.01, activation_dict: List[nn.Module]=None, linear_start=True, input_size=None, device="cpu"):
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
        self.to(device)
        self.population_size = population_size
        self.crossover_rate = int((1-crossover_rate)*population_size)
        self.weight_mutation_rate = weight_mutation_rate
        self.perturbation_rate = perturbation_rate
        self.add_neuron_mutation_rate = add_neuron_mutation_rate
        self.add_filter_mutation_rate = add_filter_mutation_rate
        self.linear_mutation_rate = linear_mutation_rate
        self.conv_mutation_rate = conv_mutation_rate
        self.activation_mutation_rate = activation_mutation_rate
        self.threshold = threshold
        
        # network parameters
        self.activation_dict = activation_functions if activation_dict is None else activation_dict
        self.linear_start = linear_start
        
        # Initialize the population
        self.population = [ATNetwork(*layers, activation=activation, last_activation=last_activation, bias=bias, input_size=input_size).to(device) for _ in range(population_size)]
        self.fitness_scores = [0.0] * population_size
        self.selection_probs = self.fitness_scores
        self.best_fitness = float("-inf")
        self.last_fitness = float("-inf")

    @torch.no_grad()
    def evaluate_fitness(self):
        """
        Evaluate the fitness of each network in the population.
        """
        self.fitness_scores = [0.0] * len(self.population)

        for i, network in enumerate(tqdm(self.population, desc=f"{BLUE}Fitness Evaluation{RESET_COLOR}", ncols=100)):
            self.fitness_scores[i] = self.fitness_fn(network)
        
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
        parent1 = random.choices(self.population, weights=self.selection_probs, k=1)[0].genotype()
        parent2 = random.choices(self.population, weights=self.selection_probs, k=1)[0].genotype()
        
        return parent1, parent2

    @torch.no_grad()
    def crossover(self, parent1: DNA, parent2: DNA) -> DNA:
        """
        Perform a more vectorized crossover between two parent networks to produce a offspring network.
        
        Args:
            parent1 (DNA): The first parent network.
            parent2 (DNA): The second parent network.
        
        Returns:
            DNA: A new offspring network created from crossover.
        """
        parent1 = copy.deepcopy(parent1)
        parent2 = copy.deepcopy(parent2)
        offspring = copy.deepcopy(parent1)
        modifier = copy.deepcopy(parent2)
        
        # Evolve layers to match neuron counts for crossover
        for i, (i1, i2) in enumerate(zip(parent1.conv_size(), parent2.conv_size())):
            if i1 < i2:
                for _ in range(i2 - i1):
                    parent1.evolve_conv_layer(i, True)
            elif i2 < i1:
                for _ in range(i1 - i2):
                    parent2.evolve_conv_layer(i, True)
                    
        for i, (i1, i2) in enumerate(zip(parent1.linear_size(), parent2.linear_size())):
            if i1 < i2:
                for _ in range(i2 - i1):
                    parent1.evolve_linear_layer(i)
            elif i2 < i1:
                for _ in range(i1 - i2):
                    parent2.evolve_linear_layer(i)

        # Determine final offspring layer sizes
        if len(parent1.conv) > len(parent2.conv):
            offspring.conv, modifier.conv = parent1.conv, parent2.conv
        else:
            offspring.conv, modifier.conv = parent2.conv, parent1.conv

        if len(parent1.linear) > len(parent2.linear):
            offspring.linear, modifier.linear = parent1.linear, parent2.linear
        else:
            offspring.linear, modifier.linear = parent2.linear, parent1.linear

        # crossover conv layers
        for idx in range(len(modifier.conv)):
            weight1 = offspring.conv[idx][0].weight
            weight2 = modifier.conv[idx][0].weight
            bias1 = offspring.conv[idx][0].bias if offspring.conv[idx][0].bias is not None else None
            bias2 = modifier.conv[idx][0].bias if modifier.conv[idx][0].bias is not None else None

            # Apply random mask to weights and biases for crossover
            mask = torch.rand_like(weight1, device=weight1.device) < 0.5
            offspring_layer_weight = torch.where(mask, weight1, weight2)
                
            if bias1 is not None and bias2 is not None:
                mask = torch.rand_like(bias1, device=weight1.device) < 0.5
                offspring_layer_bias = torch.where(mask, bias1, bias2)
            else:
                offspring_layer_bias = bias1 if bias1 is not None else bias2

            offspring.conv[idx][0].weight.data.copy_(offspring_layer_weight)
            if offspring_layer_bias is not None:
                offspring.conv[idx][0].bias.data.copy_(offspring_layer_bias)

            # Crossover activation function
            offspring.conv[idx][1].activation = offspring.conv[idx][1].activation if random.random() < 0.5 else modifier.conv[idx][1].activation

        # adjust maxpool positions
        offspring.rearrange()
            
        # crossover linear layers
        for idx in range(len(modifier.linear)):
            weight1 = offspring.linear[idx][0].weight
            weight2 = modifier.linear[idx][0].weight
            bias1 = offspring.linear[idx][0].bias if offspring.linear[idx][0].bias is not None else None
            bias2 = modifier.linear[idx][0].bias if modifier.linear[idx][0].bias is not None else None

            # Apply random mask to weights and biases for crossover
            mask = torch.rand_like(weight1, device=weight1.device) < 0.5
            offspring_layer_weight = torch.where(mask, weight1, weight2)
                
            if bias1 is not None and bias2 is not None:
                mask = torch.rand_like(bias1, device=weight1.device) < 0.5
                offspring_layer_bias = torch.where(mask, bias1, bias2)
            else:
                offspring_layer_bias = bias1 if bias1 is not None else bias2

            offspring.linear[idx][0].weight.data.copy_(offspring_layer_weight)
            if offspring_layer_bias is not None:
                offspring.linear[idx][0].bias.data.copy_(offspring_layer_bias)

            # Crossover activation function
            if idx != len(modifier.linear)-1:
                offspring.linear[idx][1] = offspring.linear[idx][1] if random.random() < 0.5 else modifier.linear[idx][1]

        return offspring

    @torch.no_grad()
    def mutate(self, network: DNA):
        """
        Mutate the given network with a small probability.
        
        Args:
            network (DNA): The network to mutate.
        """
        if random.random() > self.linear_mutation_rate:
            network.evolve_linear_network()
        if random.random() > self.conv_mutation_rate:
            network.evolve_conv_network()
        if random.random() > self.add_neuron_mutation_rate:
            network.evolve_linear_layer()
        if random.random() > self.add_filter_mutation_rate:
            network.evolve_conv_layer()
        if random.random() > self.activation_mutation_rate:
            network.evolve_activation(self.activation_dict)
        network.evolve_weight(self.weight_mutation_rate, self.perturbation_rate)

                
    def run_generation(self, metrics: int=0) -> float:
        """
        Run a single generation of the genetic algorithm, including evaluation, selection, crossover, mutation, and pruning.
        
        Args:
            fitness_fn (function): A function to evaluate the fitness of a network.
        """
        
        self.evaluate_fitness()

        # preview training data
        max_fitness = max(self.fitness_scores)
        min_fitness = min(self.fitness_scores)
        mean_fitness = numpy.mean(self.fitness_scores)
        all_fitness = [max_fitness, mean_fitness, mean_fitness]
        color = GREEN if all_fitness[metrics] > self.best_fitness else GRAY if all_fitness[metrics] > self.last_fitness else RED
        print_current_best = f"{BLUE}Best Fitness{RESET_COLOR}: \t {BOLD}{color}{ all_fitness[metrics] }{RESET_COLOR}"
        self.last_fitness = all_fitness[metrics] 
        if all_fitness[metrics] > self.best_fitness: 
            self.best_fitness = all_fitness[metrics]

        # selection of parents
        fits = sorted([fit for fit in self.fitness_scores], reverse=True)[:self.crossover_rate]
        min_fit = min(fits)
        fits = [fit-min_fit for fit in fits] # make all positive numbers
        total_fitness = sum(fits)
        self.selection_probs = [score / total_fitness for score in fits] # convert to probabilities
        
        sorted_population = [ind for _, ind in sorted(zip(self.fitness_scores, self.population), key=lambda x: x[0], reverse=True)]
        self.population = sorted_population[:self.crossover_rate]  # Select top 50%
        
        new_population: List[ATNetwork] = []
        for _ in tqdm(range(self.population_size-self.crossover_rate), desc=f"{BLUE}Crossover & Mutation{RESET_COLOR}", ncols=100):
            parent1, parent2 = self.select_parents()
            offspring = self.crossover(parent1, parent2)
            self.mutate(offspring)
            offspring = ATNetwork.phenotype(offspring)
            new_population.append(offspring)
        self.population.extend(new_population)

        print(print_current_best)
        print(f"{BLUE}{BOLD}Best{RESET_COLOR}", end=" ")
        self.population[0].summary()


        # use best genome to create experiences
        if self.is_overridden("experiences_fn"):
            self.experiences_fn(self.population[0])

        # smooth the generation networks a little back propagation based method
        if self.is_overridden("backprob_fn"):
            self.evaluate_learn()

        print_stats_table(self.best_fitness, metrics, max_fitness, mean_fitness, min_fitness, self.population_size)
        self.save_population()
        return max_fitness, mean_fitness, min_fitness

    def evolve(self, generation: int=None, fitness: int=None, save_name: str=None, metrics: int=0, plot: bool=False):
        last_fitness = -float("inf")
        maximum, mean, minimum = [], [], []
        if generation is not None:
                for i in range(generation):
                    print(f"\n--- {BLUE}Generation {i+1}{RESET_COLOR} ---")
                    results = self.run_generation(metrics)
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
                results = self.run_generation(metrics)
                maximum.append(results[0])
                mean.append(results[1])
                minimum.append(results[2])
                last_fitness = results[metrics]
            print(f"\n--- {BLUE}Fitness reached{RESET_COLOR} ---")
            
        else:
            raise ValueError("evolve using generation number or fitness value")
        
        if save_name is not None:
            self.population[0].save_network(save_name)
        
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
    ga = ATGEN(population_size=10, layers=[8, 1, 4])
    
    # Create parent networks with specified architectures
    parent1 = ATNetwork(10, 2, 5, 1)
    parent2 = ATNetwork(10, 2, 4, 3, 1)
    
    # Perform crossover
    offspring = ga.crossover(parent1.genotype(), parent2.genotype())
    offspring = ATNetwork.phenotype(offspring)
    offspring.summary()
    
    # Create CNN parent networks
    parent1 = ATNetwork(
        Conv2D(3, 32),
        ActiSwitch(nn.ReLU()),
        MaxPool2D(),
        Conv2D(32, 64),
        ActiSwitch(nn.ReLU()),
        MaxPool2D(),
        Flatten(),
        LazyLinear(100),
        ActiSwitch(nn.ReLU()),
        Linear(100, 10),
        input_size=(32, 32)
    )
    parent2 = ATNetwork(
        Conv2D(3, 32),
        ActiSwitch(nn.ReLU()),
        MaxPool2D(),
        Conv2D(32, 64),
        ActiSwitch(nn.ReLU()),
        Conv2D(64, 64),
        ActiSwitch(nn.ReLU()),
        MaxPool2D(),
        Flatten(),
        LazyLinear(200),
        ActiSwitch(nn.ReLU()),
        Linear(200, 100),
        ActiSwitch(nn.ReLU()),
        Linear(100, 10),
        input_size=(32, 32)
    )
    parent1.summary()
    parent2.summary()
    # summary(parent1, (3, 32, 32), device="cpu")
    # summary(parent2, (3, 32, 32), device="cpu")
    
    # Perform crossover
    parent2 = parent2.genotype()
    print(len(parent2.conv))
    parent2.evolve_conv_layer(1)
    offspring = ga.crossover(parent1.genotype(), parent2)
    offspring = ATNetwork.phenotype(offspring)
    offspring.summary()
    # summary(offspring, (3, 32, 32), device="cpu")

