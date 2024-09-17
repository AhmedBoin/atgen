from collections import deque
import random
import torch
from torch import nn

from atgen.ga import ATGEN
from atgen.config import ATGENConfig
from atgen.layers.activations import ActiSwitch

import gymnasium as gym
import warnings


warnings.filterwarnings("ignore", category=DeprecationWarning)

game = "CarRacing-v2"
device = "cpu"

class AutoEncoder(nn.Module):
    def __init__(self, latent_dim=10):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),  # [96, 96, 3] -> [48, 48, 16]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # [48, 48, 16] -> [24, 24, 32]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # [24, 24, 32] -> [12, 12, 64]
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(12 * 12 * 64, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 12 * 12 * 64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # [12, 12, 64] -> [24, 24, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # [24, 24, 32] -> [48, 48, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),   # [48, 48, 16] -> [96, 96, 3]
            nn.Sigmoid()
        )

    def reduce(self, x) -> torch.Tensor:
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.fc1(x)

    def forward(self, x) -> torch.Tensor:
        latent = self.reduce(x)
        x = self.fc2(latent)
        x = x.view(x.size(0), 64, 12, 12)
        x = self.decoder(x)
        return x

class NeuroEvolution(ATGEN):
    def __init__(self, population_size: int, model: nn.Sequential):
        config = ATGENConfig(crossover_rate=0.6, mutation_rate=0.8, perturbation_rate=0.9, mutation_decay=0.9, perturbation_decay=0.9, 
                             maximum_depth=3, speciation_level=1, deeper_mutation=0.01, wider_mutation=0.01, random_topology=False)
        super().__init__(population_size, model, config)
        self.autoencoder = AutoEncoder(latent_dim=10).to(device)
        try: self.autoencoder.load_state_dict(torch.load("autoencoder.pth"))
        except: pass
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=1e-3)
        self.lr_decay = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.80)
        self.buffer = deque(maxlen=10_000)
        self.my_fitness = float("-inf")
        self.steps = 50
        self.counter = 0

    @torch.no_grad()
    def fitness_fn(self, model: nn.Sequential):
        self.counter += 1
        if self.counter > self.population_size:
            if self.my_fitness < self.best_fitness:
                self.my_fitness = self.best_fitness
                self.counter = 0
                self.steps += 50
        epochs = 1
        env = gym.make(game, max_episode_steps=self.steps)
        total_reward = 0
        for _ in range(epochs):
            state, info = env.reset()
            while True:
                state = torch.FloatTensor(state/255).permute(2, 0, 1)
                self.buffer.append(state)
                feature = self.autoencoder.reduce(state.unsqueeze(0).to(self.device))
                action = model(feature).squeeze(0).detach().cpu().numpy()
                state, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                
                if terminated or truncated:
                    break
        env.close()
        print(self.steps, end="\r")
        return total_reward / epochs
    
    def pre_generation(self):
        for epoch in range(50):
            input_images = random.sample(self.buffer, 128)
            input_images = torch.stack(input_images).to(self.device)
            reconstructed_images = self.autoencoder(input_images)
            loss = self.criterion(reconstructed_images, input_images)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item()}", end="\r")
        print()
        self.lr_decay.step()
        torch.save(self.autoencoder.state_dict(), "autoencoder.pth")
    

if __name__ == "__main__":
    model = nn.Sequential(
        nn.Linear(10, 3), 
        nn.Tanh()
    ).to(device)
    ne = NeuroEvolution(50, model)
    ne.load_population()
    # ne.evolve(fitness=900, save_name="population.pkl", metrics=0, plot=True)
    
    model = ne.population.best_individual()
    env = gym.make(game, render_mode="human")
    state, info = env.reset()
    total_reward = 0
    while True:
        for individual in ne.population:
            while True:
                with torch.no_grad():
                    action = individual.model(ne.autoencoder.reduce(torch.FloatTensor(state/255).permute(2, 0, 1).unsqueeze(0).to(device))).cpu().squeeze(0).detach().numpy()
                    state, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    if terminated or truncated:
                        print(f"Last reward: {total_reward}")
                        total_reward = 0
                        state, info = env.reset()
                        break

    
