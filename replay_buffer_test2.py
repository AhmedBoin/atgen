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
        config = ATGENConfig(crossover_rate=0.8, mutation_rate=0.8, perturbation_rate=0.9, mutation_decay=0.95, perturbation_decay=0.95,
                             maximum_depth=3, default_activation=ActiSwitch(nn.Tanh(), True), speciation_level=0, deeper_mutation=0.01, wider_mutation=0.01)
        memory = ReplayBuffer(buffer_size=20, steps=50, dilation=20, threshold=0.9, patient=50, accumulative_reward=False)
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
        epochs = 20
        env = gym.make(game)
        for _ in range(epochs):
            state, info = env.reset()
            # total_reward = 0
            # steps = 0
            while True:
                # steps += 1
                state = torch.FloatTensor(state).unsqueeze(0)
                action = model(state).squeeze(0)
                next_state, reward, terminated, truncated, info = env.step(action.numpy())
                # total_reward += reward
                self.memory.track(state, action, reward)
                state = next_state
                self.memory.half_clear()
                
                if terminated or truncated:
                    break
        env.close()
        # print(f"Total reward: {total_reward}")
    

if __name__ == "__main__":
    model = nn.Sequential(nn.Linear(24, 4), nn.Tanh())
    ne = NeuroEvolution(200, model)
    NeuroEvolution.load(ne)
    ne.load("model.pkl")
    ne.load_population()
    ne.load_individual()
    ne.config = ne.config.load()
    ne.evolve(fitness=350, save_name="model.pkl", metrics=0, plot=True)
    
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
    