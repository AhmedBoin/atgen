from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F

from atgen.ga import ATGEN
from atgen.config import ATGENConfig
from atgen.layers.activations import ActiSwitch

import gymnasium as gym
import warnings


warnings.filterwarnings("ignore", category=DeprecationWarning)

game = "CarRacing-v2"
device = "mps"

import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, latent_dim=10):
        super(VAE, self).__init__()
        
        # Assuming input image size is 96x96x3
        self.input_dim = 96 * 96 * 3
        
        # Encoder: Fully connected layers
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 1024),  # Flattened input -> intermediate size
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        
        # Latent space: mean (mu) and log variance (logvar)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # Decoder: Fully connected layers
        self.fc_decode = nn.Linear(latent_dim, 256)
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.input_dim),  # Final output to match the flattened input
            nn.Sigmoid()  # Output scaled between 0 and 1 for reconstruction
        )
    
    def encode(self, x):
        # Flatten the input: [batch_size, 3, 96, 96] -> [batch_size, 3 * 96 * 96]
        x = x.reshape(x.size(0), -1)
        x = self.encoder(x)
        
        # Latent space (mean and logvar)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        # Decode from latent space back to input dimensions
        x = self.fc_decode(z)
        x = self.decoder(x)
        
        # Reshape back to image format: [batch_size, 3, 96, 96]
        x = x.reshape(x.size(0), 3, 96, 96)
        return x
    
    def forward(self, x):
        # x = x + torch.rand_like(x) * 0.5
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar

    def reduce(self, x):
        """Use the encoder to extract the latent representation (mu)."""
        mu, _ = self.encode(x)
        return mu
    

# Loss function for VAE (Reconstruction loss + KL divergence)
def vae_loss(reconstructed, original, mu, logvar):
    recon_loss = F.mse_loss(reconstructed, original, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld

class NeuroEvolution(ATGEN):
    def __init__(self, population_size: int, model: nn.Sequential):
        config = ATGENConfig(crossover_rate=0.8, mutation_rate=0.8, perturbation_rate=0.9, mutation_decay=0.9, perturbation_decay=0.9, deeper_mutation=0.01)
        super().__init__(population_size, model, config)
        self.autoencoder = VAE(latent_dim=3).to(device)
        # try: self.autoencoder.load_state_dict(torch.load("autoencoder.pth"))
        # except: pass
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=1e-3)#*(0.8**14))
        self.lr_decay = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.80)
        self.buffer = deque(maxlen=10_000)
        self.my_fitness = float("-inf")
        self.steps = 10

    @torch.no_grad()
    def fitness_fn(self, model: nn.Sequential):
        epochs = 2
        env = gym.make(game, max_episode_steps=self.steps)
        total_reward = 0
        for _ in range(epochs):
            state, info = env.reset()
            while True:
                state = torch.FloatTensor(state/255).permute(2, 0, 1)
                self.buffer.append(state)
                feature = self.autoencoder.reduce(state.unsqueeze(0).to(device))
                action = model(feature).squeeze(0).detach().cpu().numpy()
                state, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                
                if terminated or truncated:
                    break
        env.close()
        print(self.steps, end="\r")
        return total_reward / epochs
    
    def pre_generation(self):
        if self.my_fitness < self.best_fitness:
            self.my_fitness = self.best_fitness
            self.steps += 50
        else:
            self.steps += 10
        # self.steps += 50
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
    model = nn.Sequential(
        nn.Linear(3, 3), 
        nn.Tanh()
    ).to(device)
    ne = NeuroEvolution(50, model)
    # ne.load_population()
    ne.evolve(fitness=1000, save_name="population.pkl", metrics=0, plot=True)
    
    # model = ne.population.best_individual()
    env = gym.make(game, render_mode="human")
    state, info = env.reset()
    total_reward = 0
    while True:
        for i, individual in enumerate(ne.population):
            # individual = ne.population[12]
            while True:
                with torch.no_grad():
                    action = individual.model(ne.autoencoder.reduce(torch.FloatTensor(state/255).permute(2, 0, 1).unsqueeze(0).to(device))).cpu().squeeze(0).detach().numpy()
                    state, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    if terminated or truncated:
                        print(f"{i} reward: {total_reward}")
                        total_reward = 0
                        state, info = env.reset()
                        break

    
