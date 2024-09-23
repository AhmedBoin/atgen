import torch
from torch import nn

from atgen.ga import ATGEN
from atgen.config import ATGENConfig
from atgen.layers.activations import ActiSwitch

import gymnasium as gym
import warnings


warnings.filterwarnings("ignore", category=DeprecationWarning)

game = "LunarLander-v2"

class NeuroEvolution(ATGEN):
    def __init__(self, population_size: int, model: nn.Sequential):
        config = ATGENConfig(crossover_rate=0.8, mutation_rate=0.8, perturbation_rate=0.9, mutation_decay=0.9, perturbation_decay=0.9, speciation_level=1, deeper_mutation=0.00)
        super().__init__(population_size, model, config)

    @torch.no_grad()
    def fitness_fn(self, model: nn.Sequential):
        epochs = 3
        env = gym.make(game, max_episode_steps=2000)
        total_reward = 0
        for _ in range(epochs):
            state, info = env.reset()
            while True:
                action = model(torch.FloatTensor(state).unsqueeze(0)).argmax().item()
                next_state, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                state = next_state
                
                if terminated or truncated:
                    break
        env.close()
        return total_reward / epochs
    

if __name__ == "__main__":
    model = nn.Sequential(nn.Linear(8, 4))
    ne = NeuroEvolution(300, model)
    ne.evolve(fitness=280, save_name="population.pkl", metrics=0, plot=True)
    
    model = ne.population.best_individual()
    env = gym.make(game, render_mode="human")
    state, info = env.reset()
    total_reward = 0
    while True:
        with torch.no_grad():
            action = model(torch.FloatTensor(state).unsqueeze(0)).argmax().item()
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                print(f"Last reward: {total_reward}")
                total_reward = 0
                state, info = env.reset()
    
