import torch
from torch import nn

from atgen.ga import ATGEN
from atgen.config import ATGENConfig
from atgen.memory import ReplayBuffer
from atgen.layers.activations import ActiSwitch

import gymnasium as gym
import warnings



warnings.filterwarnings("ignore", category=DeprecationWarning)

game = "LunarLander-v2"

class NeuroEvolution(ATGEN):
    def __init__(self, population_size: int, model: nn.Sequential):
        config = ATGENConfig(crossover_rate=0.8, mutation_rate=0.8, perturbation_rate=0.9, mutation_decay=0.9, 
                             perturbation_decay=0.9, speciation_level=1, deeper_mutation=0.00, elitism=True)
        
        memory = ReplayBuffer(buffer_size=20, steps=50, dilation=20, discrete_action=True, threshold=0.9, # 65
                              prioritize=True, accumulative_reward=True, patient=50) # 10 reward_range=(-99, 99), 
        
        super().__init__(population_size, model, config, memory)

    @torch.no_grad()
    def fitness_fn(self, model: nn.Sequential):
        epochs = 10
        env = gym.make(game, max_episode_steps=2000)
        total_reward = 0
        for _ in range(epochs):
            state, info = env.reset()
            while True:
                action = model(torch.FloatTensor(state).unsqueeze(0)).argmax().item()
                next_state, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                state = next_state
                self.memory.half_clear()
                
                if terminated or truncated:
                    break
        env.close()
        return total_reward / epochs
    

    @torch.no_grad()
    def experiences_fn(self, model: nn.Sequential):
        epochs = 10
        env = gym.make(game)
        total_reward = 0
        for _ in range(epochs):
            state, info = env.reset()
            while True:
                state = torch.FloatTensor(state).unsqueeze(0)
                action = model(state).argmax()
                next_state, reward, terminated, truncated, info = env.step(action.item())
                total_reward += reward
                self.memory.track(state, action, total_reward)
                state = next_state
                
                if terminated or truncated:
                    total_reward = 0
                    break
        env.close()
    

if __name__ == "__main__":
    model = nn.Sequential(nn.Linear(8, 4))
    ne = NeuroEvolution(50, model)
    # ne.load_population()
    # ne.load_individual()
    ne.evolve(fitness=280, save_name="population.pkl", metrics=0, plot=True)
    
    model = ne.population.best_individual()
    env = gym.make(game, render_mode="human")
    state, info = env.reset()
    total_reward = 0
    steps = 0
    while True:
        with torch.no_grad():
            action = model(torch.FloatTensor(state).unsqueeze(0)).argmax().item()
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                print(f"{steps} Last reward: {total_reward}")
                steps = 0
                total_reward = 0
                state, info = env.reset()
    
