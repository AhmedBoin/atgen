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
        config = ATGENConfig(maximum_depth=1, elitism=False)
        memory = ReplayBuffer(buffer_size=4, steps=50, dilation=20, discrete_action=True, accumulative_reward=True, similarity_cohort=50)
        
        super().__init__(population_size, model, config, memory)

    @torch.no_grad()
    def fitness_fn(self, model: nn.Sequential):
        env = gym.make(game)
        total_reward = 0
        state, info = env.reset()
        epochs = 5
        for i in range(epochs):
            while True:
                state = torch.FloatTensor(state).unsqueeze(0)
                action = model(state).argmax()
                next_state, reward, terminated, truncated, info = env.step(action.item())
                total_reward += reward
                self.memory.track(state, action, total_reward)
                state = next_state
                
                if terminated or truncated:
                    state, info = env.reset()
                    self.memory.new()
                    break
        env.close()
        return total_reward / epochs
    

if __name__ == "__main__":
    model = nn.Sequential(nn.Linear(8, 4))
    ne = NeuroEvolution(500, model)
    # ne.load("LunarLander")
    ne.evolve(fitness=250, log_name="LunarLander", metrics=0, plot=True)
    
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
    
