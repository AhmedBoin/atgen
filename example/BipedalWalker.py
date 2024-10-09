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
        config = ATGENConfig(mutation_decay=0.95, perturbation_decay=0.95, maximum_depth=1, deeper_mutation=0.01)
        memory = ReplayBuffer(buffer_size=4, steps=70, dilation=20, similarity_cohort=50, accumulative_reward=True)
        super().__init__(population_size, model, config, memory)

    @torch.no_grad()
    def fitness_fn(self, model: nn.Sequential):
        env = gym.make(game)
        state, info = env.reset()
        total_reward = 0
        while True:
            state = torch.FloatTensor(state).unsqueeze(0)
            action = model(state).squeeze(0)
            next_state, reward, terminated, truncated, info = env.step(action.numpy())
            total_reward += reward
            self.memory.track(state, action, total_reward)
            state = next_state
            
            if terminated or truncated:
                self.memory.new()
                break
        env.close()
        return total_reward
    

if __name__ == "__main__":
    model = nn.Sequential(nn.Linear(24, 4), nn.Tanh())
    ne = NeuroEvolution(200, model)
    # ne.load("BipedalWalker")
    ne.evolve(fitness=300, log_name="BipedalWalker", metrics=0, plot=True)
    
    model = ne.population.best_individual()
    env = gym.make(game, render_mode="human")
    state, info = env.reset()
    total_reward = 0
    while True:
        with torch.no_grad():
            action = model(torch.FloatTensor(state).unsqueeze(0)).squeeze(0).numpy()
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            print(f"Last reward: {total_reward}")
            if terminated or truncated:
                total_reward = 0
                state, info = env.reset()
    
