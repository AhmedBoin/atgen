import torch
import torch.nn.functional as F
from torch.optim import AdamW

from atgen.layers.activations import Pass
from atgen.network import ATNetwork
from atgen.ga import ATGEN

import gymnasium as gym
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

env = gym.make("LunarLander-v2")

class NeuroEvolution(ATGEN):
    def __init__(self, population_size: int, layers: F.List[int]):
        super().__init__(population_size, layers, activation=torch.nn.ReLU, last_activation=Pass(), batch_size=128, backprob_phase=True, experiences_phase=True, 
                         weight_mutation_rate=0.3, perturbation_rate=0.3, layer_mutation_rate=0, network_mutation_rate=0, activation_mutation_rate=0)

    def fitness_fn(self, model: ATNetwork):
        epochs = 10
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

    @torch.no_grad()
    def experiences_fn(self, model: ATNetwork):
        epochs = 10
        gamma = 0.80  # Set gamma to a suitable value for discounting future rewards
        env = gym.make("LunarLander-v2")
        
        for _ in range(epochs):
            state, info = env.reset()
            episode_experiences = []  # To store experiences in the current episode
            
            while True:
                action = model(torch.FloatTensor(state).unsqueeze(0)).argmax().item()
                next_state, reward, terminated, truncated, info = env.step(action)
                
                # Store the experience temporarily in the episode_experiences list
                episode_experiences.append((state, action, reward, next_state, terminated or truncated))
                
                state = next_state
                if terminated or truncated:
                    break
            
            # Reverse iterate over episode experiences to calculate discounted rewards
            cumulative_reward = 0
            for experience in reversed(episode_experiences):
                state, action, reward, next_state, done = experience
                cumulative_reward = reward + gamma * cumulative_reward
                # Add the experience with the discounted reward to the memory
                self.memory.add(state, action, cumulative_reward, next_state, done)
        
        env.close()
    
    
    def backprob_fn(self, model: ATNetwork):
        epochs = 10
        gamma = 0.99
        optimizer = AdamW(model.parameters(), lr=1e-2)
        for i in range(epochs):
            states, actions, rewards, next_states, dones = self.memory.sample()
            
            Q_targets_next = model(next_states).detach().max(dim=1, keepdim=True)[0]
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
            
            Q_expected = model(states).gather(dim=1, index=actions)
            
            loss = F.huber_loss(Q_expected, Q_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # if i == (epochs // 2):
            #     model.prune()
            #     optimizer = AdamW(model.parameters(), lr=1e-1)

if __name__ == "__main__":
    ne = NeuroEvolution(100, [8, 32, 32, 4])
    ne.evolve(fitness=250, save_name="ATNetwork.pth")
    
    model = ATNetwork.load_network()
    env = gym.make("LunarLander-v2")#, render_mode="human")
    state, info = env.reset()
    total_reward = 0
    while True:
        with torch.no_grad():
            action = model(torch.FloatTensor(state).unsqueeze(0)).argmax().item()
            next_state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                print("Last reward:", total_reward)
                total_reward = 0
                state, info = env.reset()
    
