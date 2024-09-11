import random
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from atgen.layers.activations import Pass
from atgen.network import ATNetwork
from atgen.ga import ATGEN
from atgen.config import ATGENConfig

import gymnasium as gym
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

env = gym.make("LunarLander-v2")

class NeuroEvolution(ATGEN):
    def __init__(self, population_size: int, layers: F.List[int]):
        super().__init__(population_size, layers, ATGENConfig(last_mutation_rate=0.01, neuron_mutation_rate=0.02))

    def fitness_fn(self, model: ATNetwork):
        epochs = 3
        env = gym.make("LunarLander-v2")
        total_reward = 0
        for _ in range(epochs):
            state, info = env.reset()
            while True:
                with torch.no_grad():
                    action = model(torch.FloatTensor(state).unsqueeze(0)).argmax().item()
                next_state, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                state = next_state
                
                if terminated or truncated:
                    break
        env.close()
        # print("reword:", total_reward / epochs, end="\r")
        return total_reward / epochs
    
    # @torch.no_grad()
    # def experiences_fn(self, model: ATNetwork):
    #     epochs = 10
    #     env = gym.make("LunarLander-v2")
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
    # def experiences_fn(self, model: ATNetwork):
    #     epochs = 10
    #     gamma = 0.80  # Set gamma to a suitable value for discounting future rewards
    #     env = gym.make("LunarLander-v2")
        
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
    
    
    # def backprob_fn(self, model: ATNetwork):
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
    ne = NeuroEvolution(1000, [8, 4])
    ne.load_population()
    # ne.evolve(fitness=280, save_name="ATNetwork.pth", metrics=0, plot=True)
    
    # model = ATNetwork.load_network()
    model = ne.population[1]
    # env = gym.make("LunarLander-v2", render_mode="human")
    while True:
        # for i, model in enumerate(ne.population[:ne.crossover_rate]):
            env = gym.make("LunarLander-v2", render_mode="human")
            state, info = env.reset()
            total_reward = 0
            while True:
                with torch.no_grad():
                    action = model(torch.FloatTensor(state).unsqueeze(0)).argmax().item()
                    state, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    if terminated or truncated:
                        print(f"Last reward: {total_reward}")
                        # print(f"model: {i:<15}Last reward: {total_reward}")
                        total_reward = 0
                        state, info = env.reset()
                        break
    
