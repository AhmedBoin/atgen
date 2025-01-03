import torch
from torch import nn

from atgen.ga import ATGEN
from atgen.config import ATGENConfig
from atgen.memory import ReplayBuffer
from atgen.layers.activations import ActiSwitch

import gymnasium as gym
import warnings


warnings.filterwarnings("ignore", category=DeprecationWarning)

game = "Pendulum-v1"

class NeuroEvolution(ATGEN):
    def __init__(self, population_size: int, model: nn.Sequential):
        config = ATGENConfig(deeper_mutation=0.01, linear_start=False, patience=1)
        super().__init__(population_size, model, config)

    @torch.no_grad()
    def fitness_fn(self, model: nn.Sequential):
        env = gym.make(game)
        state, info = env.reset()
        total_reward = 0
        while True:
            state = torch.FloatTensor(state).unsqueeze(0)
            action = model(state).squeeze(0)
            next_state, reward, terminated, truncated, info = env.step(action.numpy() * 2)
            total_reward += reward
            state = next_state
            
            if terminated or truncated:
                break
        env.close()
        return total_reward
    

if __name__ == "__main__":
    model = nn.Sequential(nn.Linear(3, 1), nn.Tanh())
    ne = NeuroEvolution(500, model)
    # ne.load("Pendulum")
    ne.evolve(fitness=-0.5, log_name="Pendulum", metrics=0, plot=True)
    
    model = ne.population.best_individual()
    env = gym.make(game, render_mode="human")
    state, info = env.reset()
    total_reward = 0
    while True:
        with torch.no_grad():
            action = model(torch.FloatTensor(state).unsqueeze(0)).squeeze(0).numpy()
            state, reward, terminated, truncated, info = env.step(action * 2)
            total_reward += reward
            print(f"Last reward: {total_reward}")
            if terminated or truncated:
                total_reward = 0
                state, info = env.reset()
    
