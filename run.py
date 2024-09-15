import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW

from atgen.ga import ATGEN
from atgen.memory import ReplayBuffer
from atgen.config import ATGENConfig

import gymnasium as gym
import warnings

from atgen.layers.activations import ActiSwitch

warnings.filterwarnings("ignore", category=DeprecationWarning)

game = "BipedalWalker-v3"

class NeuroEvolution(ATGEN):
    def __init__(self, population_size: int, model: nn.Sequential):
        config = ATGENConfig(crossover_rate=0.8, mutation_rate=0.03, perturbation_rate=0.02, log_level=0, maximum_depth=3,
                             single_offspring=False, speciation_level=1, deeper_mutation=0.01, wider_mutation=0.1, random_topology=True)
        super().__init__(population_size, model, config)
        # self.memory = ReplayBuffer(24, 5000)
        self.steps = 100
        self.my_fitness = float("-inf")

    def fitness_fn(self, model: nn.Sequential):
        if self.best_fitness > self.my_fitness:
            self.my_fitness = self.best_fitness
            self.steps += 100
        epochs = 1
        env = gym.make(game)
        total_reward = 0
        for _ in range(epochs):
            state, info = env.reset()
            for _ in range(self.steps):
                with torch.no_grad():
                    # action = model(torch.FloatTensor(state).unsqueeze(0)).argmax().item()     # lunar-lander
                    action = model(torch.FloatTensor(state).unsqueeze(0)).squeeze(0).numpy()
                state, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                # state = next_state
                
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
    #             # action = model(torch.FloatTensor(state).unsqueeze(0)).argmax().item()
    #             action = model(torch.FloatTensor(state).unsqueeze(0)).squeeze(0).numpy()
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
    #     gamma = 0.9
    #     optimizer = AdamW(model.parameters(), lr=1e-3)
    #     for _ in range(epochs):
    #         states, actions, rewards, next_states, dones = self.memory.sample()
            
    #         # Q_targets_next = model(next_states).detach().max(dim=1, keepdim=True)[0]
    #         Q_targets_next = model(next_states).detach()
    #         Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
            
    #         # Q_expected = model(states).gather(dim=1, index=actions)
    #         Q_expected = model(states)
            
    #         loss = F.huber_loss(Q_expected, Q_targets)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

if __name__ == "__main__":
    model = nn.Sequential(nn.Linear(24, 4), nn.Tanh())
    ne = NeuroEvolution(1000, model)
    # ne.config = ne.config.load()
    # ne.load_population()
    ne.evolve(fitness=280, save_name="population.pkl", metrics=0, plot=True)
    # ne.evolve(generation=1, save_name="population.pkl", metrics=0, plot=True)
    
    model = ne.best_individual
    env = gym.make(game, render_mode="human")
    while True:
        # for i, model in enumerate(ne.population.values()):
            # model = model[i][0]
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
    
