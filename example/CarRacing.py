import torch
from torch import nn

from atgen.ga import ATGEN
from atgen.config import ATGENConfig
from atgen.layers.activations import ActiSwitch

import gymnasium as gym
import warnings


warnings.filterwarnings("ignore", category=DeprecationWarning)

game = "CarRacing-v2"

class NeuroEvolution(ATGEN):
    def __init__(self, population_size: int, model: nn.Sequential):
        config = ATGENConfig(crossover_rate=0.8, mutation_rate=0.8, perturbation_rate=0.9, mutation_decay=0.9, perturbation_decay=0.9, 
                             maximum_depth=3, speciation_level=1, deeper_mutation=0.01, wider_mutation=0.01, random_topology=False)
        super().__init__(population_size, model, config)
        self.my_fitness = float("-inf")
        self.steps = 50
        self.counter = 0

    @torch.no_grad()
    def fitness_fn(self, model: nn.Sequential):
        self.counter += 1
        if self.counter > self.population_size:
            if self.my_fitness < self.best_fitness:
                self.my_fitness = self.best_fitness
                self.counter = 0
                self.steps += 50
        epochs = 1
        env = gym.make(game, max_episode_steps=self.steps)
        total_reward = 0
        for _ in range(epochs):
            state, info = env.reset()
            while True:
                action = model(torch.FloatTensor(state/255).permute(2, 0, 1).unsqueeze(0).to("mps")).cpu().squeeze(0).detach().numpy()
                state, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                # state = next_state
                
                if terminated or truncated:
                    break
        env.close()
        print(self.steps, end="\r")
        return total_reward / epochs
    

if __name__ == "__main__":
    model = nn.Sequential(
        nn.Conv2d(3, 3, 3, 3),
        ActiSwitch(nn.ReLU()),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(3, 3, 3, 3),
        ActiSwitch(nn.ReLU()),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.LazyLinear(8), 
        ActiSwitch(nn.ReLU()),
        nn.LazyLinear(3), 
        nn.Tanh()
    ).to("mps")
    model(torch.rand(1, 3, 96, 96).to("mps")).cpu().squeeze(0).detach().numpy()
    ne = NeuroEvolution(50, model)
    ne.evolve(fitness=500, save_name="population.pkl", metrics=0, plot=True)
    
    model = ne.population.best_individual()
    env = gym.make(game, render_mode="human")
    state, info = env.reset()
    total_reward = 0
    while True:
        with torch.no_grad():
            action = model(torch.FloatTensor(state/255).permute(2, 0, 1).unsqueeze(0).to("mps")).cpu().squeeze(0).detach().numpy()
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                print(f"Last reward: {total_reward}")
                total_reward = 0
                state, info = env.reset()
    
