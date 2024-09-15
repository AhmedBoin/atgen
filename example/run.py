import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW

from atgen.ga import ATGEN
from atgen.config import ATGENConfig

import gymnasium as gym
import warnings

from atgen.layers.activations import ActiSwitch

warnings.filterwarnings("ignore", category=DeprecationWarning)

env = gym.make("LunarLander-v2")

class NeuroEvolution(ATGEN):
    def __init__(self, population_size: int, model: nn.Sequential):
        config = ATGENConfig(crossover_rate=0.8, mutation_rate=0.8, perturbation_rate=0.9, mutation_decay=0.9, log_level=0,
                             perturbation_decay=0.9, crossover_decay=0.999, single_offspring=False, shared_fitness=False)
        super().__init__(population_size, model, config)

    def fitness_fn(self, model: nn.Sequential):
        epochs = 2
        steps = 1600
        # env = gym.make("LunarLander-v2")
        total_reward = 0
        for _ in range(epochs):
            state, info = env.reset()
            for _ in range(steps):
                with torch.no_grad():
                    action = model(torch.FloatTensor(state).unsqueeze(0)).argmax().item()     # lunar-lander
                next_state, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                state = next_state
                
                if terminated or truncated:
                    break
        env.close()
        return total_reward / epochs
    
    # @torch.no_grad()
    # def experiences_fn(self, model: nn.Sequential):
    #     epochs = 10
    #     env = gym.make(game)
    #     for _ in range(epochs):
    #         state, info = env.reset()
    #         while True:
    #             action = model(torch.FloatTensor(state).unsqueeze(0)).argmax().item()
    #             next_state, reward, terminated, truncated, info = env.step(action)
    #             self.memory.add(state, action, reward, next_state, terminated or truncated)
    #             state = next_state
                
    #             if terminated or truncated:
    #                 break
    #     env.close()

    # @torch.no_grad()
    # def experiences_fn(self, model: nn.Sequential):
    #     epochs = 10
    #     gamma = 0.80  # Set gamma to a suitable value for discounting future rewards
    #     env = gym.make(game)
        
    #     for _ in range(epochs):
    #         state, info = env.reset()
    #         episode_experiences = []  # To store experiences in the current episode
            
    #         while True:
    #             action = model(torch.FloatTensor(state).unsqueeze(0)).argmax().item()
    #             action = action if random.random() > 0.5 else random.randint(0, 3)
    #             next_state, reward, terminated, truncated, info = env.step(action)
                
    #             # Store the experience temporarily in the episode_experiences list
    #             episode_experiences.append((state, action, reward, next_state, terminated or truncated))
                
    #             state = next_state
    #             if terminated or truncated:
    #                 break
            
    #         # Reverse iterate over episode experiences to calculate discounted rewards
    #         cumulative_reward = 0
    #         for experience in reversed(episode_experiences):
    #             state, action, reward, next_state, done = experience
    #             cumulative_reward = reward + gamma * cumulative_reward
    #             # Add the experience with the discounted reward to the memory
    #             self.memory.add(state, action, cumulative_reward, next_state, done)
        
    #     env.close()
    
    
    # def backprob_fn(self, model: nn.Sequential):
    #     epochs = 100
    #     gamma = 0.99
    #     optimizer = AdamW(model.parameters(), lr=1e-4)
    #     for i in range(epochs):
    #         states, actions, rewards, next_states, dones = self.memory.sample()
            
    #         Q_targets_next = model(next_states).detach().max(dim=1, keepdim=True)[0]
    #         Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
            
    #         Q_expected = model(states).gather(dim=1, index=actions)
            
    #         loss = F.huber_loss(Q_expected, Q_targets)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
            
    #         # if i == (epochs // 2):
    #         #     model.prune()
    #         #     optimizer = AdamW(model.parameters(), lr=1e-1)

if __name__ == "__main__":
    model = nn.Sequential(nn.Linear(24, 16), ActiSwitch(nn.Tanh()), nn.Linear(16, 16), ActiSwitch(nn.Tanh()), nn.Linear(16, 4), nn.Tanh())
    ne = NeuroEvolution(100, model)
    # ne.load_population()
    ne.evolve(fitness=280, save_name="population.pkl", metrics=0, plot=True)
    
    model = ne.best_individual
    env = gym.make("LunarLander-v2", render_mode="human")
    while True:
        # for i, model in enumerate(ne.population.values()):
            # model = model[i][0]
            # env = gym.make("LunarLander-v2", render_mode="human")
            state, info = env.reset()
            total_reward = 0
            while True:
                with torch.no_grad():
                    # action = model(torch.FloatTensor(state).unsqueeze(0)).argmax().item()
                    action = model(torch.FloatTensor(state).unsqueeze(0)).squeeze(0).numpy()
                    state, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    if terminated or truncated:
                        print(f"Last reward: {total_reward}")
                        # print(f"model: {i:<15}Last reward: {total_reward}")
                        total_reward = 0
                        state, info = env.reset()
                        break
    
