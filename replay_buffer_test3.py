from collections import deque
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from atgen.ga import ATGEN
from atgen.config import ATGENConfig
from atgen.layers.activations import ActiSwitch

from PIL import Image

import gymnasium as gym
import warnings

from atgen.memory import ReplayBuffer


warnings.filterwarnings("ignore", category=DeprecationWarning)

game = "CarRacing-v2"
device = "mps"

class VAE(nn.Module):
    def __init__(self, latent_dim=10):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),  # [96, 96, 3] -> [48, 48, 16]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # [48, 48, 16] -> [24, 24, 32]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # [24, 24, 32] -> [12, 12, 64]
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(12 * 12 * 64, latent_dim)
        self.fc_logvar = nn.Linear(12 * 12 * 64, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 12 * 12 * 64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # [12, 12, 64] -> [24, 24, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # [24, 24, 32] -> [48, 48, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),   # [48, 48, 16] -> [96, 96, 3]
            nn.Sigmoid()
        )
    
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(x.size(0), 64, 12, 12)
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar

    def reduce(self, x):
        """Use the encoder to extract the latent representation (mu)."""
        mu, _ = self.encode(x)
        return mu.view(1, -1)

# Loss function for VAE (Reconstruction loss + KL divergence)
def vae_loss(reconstructed, original, mu, logvar):
    recon_loss = F.mse_loss(reconstructed, original, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld

class NeuroEvolution(ATGEN):
    def __init__(self, population_size: int, model: nn.Sequential):
        config = ATGENConfig(deeper_mutation=1, difficulty=3)
        memory = ReplayBuffer(buffer_size=5, steps=20, dilation=0, similarity_cohort=50, accumulative_reward=True, prioritize=True, patience=2)
        super().__init__(population_size, model, config, memory)
        self.autoencoder = VAE(latent_dim=3).to(device)
        # try: self.autoencoder.load_state_dict(torch.load("autoencoder.pth"))
        # except: pass
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=1e-3)
        self.lr_decay = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.80)
        self.buffer = deque(maxlen=10_000)
        self.my_fitness = float("-inf")
        self.steps = 10

    @torch.no_grad()
    def fitness_fn(self, model: nn.Sequential):
        env = gym.make(game, max_episode_steps=self.steps)
        total_reward = 0
        state, info = env.reset()
        while True:
            state = torch.FloatTensor(state/255).permute(2, 0, 1)
            self.buffer.append(state)
            feature = self.autoencoder.reduce(state.unsqueeze(0).to(device))
            track_action = model(feature)
            action = track_action.squeeze(0).detach().cpu().numpy()
            state, reward, terminated, truncated, info = env.step(action)
            tracked_reward = reward if reward > 0 else (- np.abs(tracked_reward))
            self.memory.track(feature, track_action, tracked_reward)
            total_reward += reward
            
            if terminated or truncated:
                break
        env.close()
        print(self.steps, end="\r")
        return total_reward
    
    def pre_generation(self):
        if self.my_fitness < self.best_fitness:
            self.my_fitness = self.best_fitness
            self.steps += 50
        else:
            self.steps += 10
        if self.steps > 900:
            self.steps = 900
        for epoch in range(50):
            input_images = random.sample(self.buffer, 128)
            input_images = torch.stack(input_images).to(device)
            reconstructed, mu, logvar = self.autoencoder(input_images)
            loss = vae_loss(reconstructed, input_images, mu, logvar)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item()}", end="\r")
        print()
        self.lr_decay.step()
        torch.save(self.autoencoder.state_dict(), "autoencoder.pth")
    

if __name__ == "__main__":
    model = nn.Sequential(nn.Linear(3, 3), nn.Tanh()).to(device)
    ne = NeuroEvolution(50, model)
    # ne.load("CarRacing")
    ne.evolve(fitness=900, log_name="CarRacing", metrics=0, plot=True)
    
    
    env = gym.make(game, render_mode="human")
    state, info = env.reset()
    total_reward = 0
    while True:
        for i, individual in enumerate(ne.population):
            individual = ne.best_individual
            steps = 0
            while True:
                steps += 1
                with torch.no_grad():
                    action = individual.model(ne.autoencoder.reduce(torch.FloatTensor(state/255).permute(2, 0, 1).unsqueeze(0).to(device))).cpu().squeeze(0).detach().numpy()
                    state, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    if terminated or truncated:
                        print(f"{i} reward: {total_reward} in {steps} steps")
                        total_reward = 0
                        state, info = env.reset()
                        break

    
