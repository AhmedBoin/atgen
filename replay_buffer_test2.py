import torch
from torch import nn

from atgen.ga import ATGEN
from atgen.config import ATGENConfig
from atgen.memory import ReplayBuffer
from atgen.layers.activations import ActiSwitch

import gymnasium as gym
import warnings



warnings.filterwarnings("ignore", category=DeprecationWarning)

game = "BipedalWalker-v3"

class NeuroEvolution(ATGEN):
    def __init__(self, population_size: int, model: nn.Sequential):
        config = ATGENConfig(crossover_rate=0.8, mutation_rate=0.8, perturbation_rate=0.9, mutation_decay=0.9, 
                             perturbation_decay=0.9, speciation_level=1, deeper_mutation=0.00, parent_mutation=False)
        memory = ReplayBuffer(buffer_size=40, steps=20, discrete_action=False, threshold=0.05, method="zscore", 
                              scale_range=(-1, 1), reward_range=(-1, 1), average_distance=True, prioritize=False)
        super().__init__(population_size, model, config, memory)

    @torch.no_grad()
    def fitness_fn(self, model: nn.Sequential):
        epochs = 1
        env = gym.make(game)
        total_reward = 0
        for _ in range(epochs):
            state, info = env.reset()
            while True:
                action = model(torch.FloatTensor(state).unsqueeze(0)).squeeze(0).numpy()
                state, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                
                if terminated or truncated:
                    break
        env.close()
        return total_reward / epochs
    
    @torch.no_grad()
    def experiences_fn(self, model: nn.Sequential):
        epochs = 5
        env = gym.make(game)
        for _ in range(epochs):
            state, info = env.reset()
            while True:
                state = torch.FloatTensor(state).unsqueeze(0)
                action = model(state).squeeze(0)
                next_state, reward, terminated, truncated, info = env.step(action.numpy())
                self.memory.track(state, action, reward)
                state = next_state
                
                if terminated or truncated:
                    break
        env.close()
    

if __name__ == "__main__":
    model = nn.Sequential(nn.Linear(24, 4))
    ne = NeuroEvolution(20, model)
    ne.evolve(fitness=290, save_name="population.pkl", metrics=0, plot=True)
    
    model = ne.population.best_individual()
    env = gym.make(game, render_mode="human")
    state, info = env.reset()
    total_reward = 0
    while True:
        with torch.no_grad():
            action = model(torch.FloatTensor(state).unsqueeze(0)).squeeze(0).numpy()
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                print(f"Last reward: {total_reward}")
                total_reward = 0
                state, info = env.reset()
    
