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
        memory = ReplayBuffer(buffer_size=50, steps=70, dilation=20, similarity_cohort=50, similarity_threshold=0.7)
        super().__init__(population_size, model, config, memory)

    @torch.no_grad()
    def fitness_fn(self, model: nn.Sequential):
        env = gym.make(game)
        state, info = env.reset()
        total_reward = 0
        while True:
            action = model(torch.FloatTensor(state).unsqueeze(0)).squeeze(0).numpy()
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        env.close()
        return total_reward
    
    @torch.no_grad()
    def experiences_fn(self, model: nn.Sequential):
        epochs = 5
        env = gym.make(game, max_episode_steps=self.steps)
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
    model = nn.Sequential(nn.Linear(24, 4), nn.Tanh())
    ne = NeuroEvolution(200, model)
    # ne.load("BipedalWalker")
    ne.evolve(fitness=300, save_name="BipedalWalker", metrics=0, plot=True)
    
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
    
