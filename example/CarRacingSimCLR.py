from collections import deque
import random

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision.transforms import functional as TF

from atgen.ga import ATGEN
from atgen.config import ATGENConfig
from atgen.layers.activations import ActiSwitch

import gymnasium as gym
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

game = "CarRacing-v2"
device = "cpu"

class SimCLR(nn.Module):
    def __init__(self, latent_dim=10):
        super(SimCLR, self).__init__()
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

        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, x) -> torch.Tensor:
        features = self.extract_features(x)
        projection = self.projection_head(features)
        return features, projection

    def extract_features(self, x):
        """Extract features (without projection head)"""
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.fc1(x)

class NeuroEvolution(ATGEN):
    def __init__(self, population_size: int, model: nn.Sequential):
        config = ATGENConfig(crossover_rate=0.6, mutation_rate=0.8, perturbation_rate=0.9, mutation_decay=0.9, perturbation_decay=0.9, 
                             maximum_depth=3, speciation_level=1, deeper_mutation=0.01, wider_mutation=0.01, random_topology=False)
        super().__init__(population_size, model, config)
        self.simclr = SimCLR(latent_dim=10).to(device)
        try: self.simclr.load_state_dict(torch.load("simclr.pth"))
        except: pass
        self.optimizer = torch.optim.Adam(self.simclr.parameters(), lr=1e-3)
        self.lr_decay = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.80)
        self.buffer = deque(maxlen=10_000)
        self.my_fitness = float("-inf")
        self.steps = 50

        # Data augmentations for SimCLR
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(96),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter()], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor()
        ])

    @torch.no_grad()
    def fitness_fn(self, model: nn.Sequential):
        epochs = 1
        env = gym.make(game, max_episode_steps=self.steps)
        total_reward = 0
        for _ in range(epochs):
            state, info = env.reset()
            while True:
                state = torch.FloatTensor(state / 255).permute(2, 0, 1)
                self.buffer.append(state)
                feature = self.simclr.extract_features(state.unsqueeze(0).to(device))
                action = model(feature).squeeze(0).detach().cpu().numpy()
                state, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                
                if terminated or truncated:
                    break
        env.close()
        print(self.steps, end="\r")
        return total_reward / epochs
    
    def contrastive_loss(self, z_i, z_j, temperature=0.5):
        """NT-Xent contrastive loss."""
        batch_size = z_i.size(0)

        # Normalize projections
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Positive pair similarity (between z_i and z_j)
        positive_sim = torch.exp(torch.sum(z_i * z_j, dim=1) / temperature)

        # Compute negative similarity
        negatives = torch.cat([z_i, z_j], dim=0)  # Concatenate z_i and z_j along batch dimension
        sim_matrix = torch.exp(torch.mm(negatives, negatives.T) / temperature)  # Similarity matrix

        # Create a mask to remove similarity between identical samples
        mask = ~torch.eye(2 * batch_size, device=z_i.device).bool()

        # Select negative similarities
        negative_sim = sim_matrix.masked_select(mask).view(2 * batch_size, -1).sum(dim=1)

        # Split negative similarity to match positive similarity size
        negative_sim_i = negative_sim[:batch_size]  # First half (negative sim for z_i)
        negative_sim_j = negative_sim[batch_size:]  # Second half (negative sim for z_j)

        # Combine the two
        negative_sim_combined = negative_sim_i + negative_sim_j

        # Compute loss based on positive to negative ratio
        loss = -torch.log(positive_sim / negative_sim_combined).mean()
        
        return loss

    def pre_generation(self):
        if self.my_fitness < self.best_fitness:
            self.my_fitness = self.best_fitness
            self.steps += 50
        for epoch in range(50):
            # Get pairs of augmented images from buffer
            input_images = random.sample(self.buffer, 128)
            input_images = torch.stack(input_images).to(device)
            
            # Convert tensors to PIL images, apply augmentation, and convert back to tensors
            aug_images_1 = torch.stack([self.transform(TF.to_pil_image(img.cpu())) for img in input_images]).to(device)
            aug_images_2 = torch.stack([self.transform(TF.to_pil_image(img.cpu())) for img in input_images]).to(device)

            # Get features and projections
            _, proj_1 = self.simclr(aug_images_1)
            _, proj_2 = self.simclr(aug_images_2)

            # Calculate contrastive loss
            loss = self.contrastive_loss(proj_1, proj_2)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item()}", end="\r")
        print()
        self.lr_decay.step()
        torch.save(self.simclr.state_dict(), "simclr.pth")


if __name__ == "__main__":
    model = nn.Sequential(
        nn.Linear(10, 3), 
        nn.Tanh()
    ).to(device)
    ne = NeuroEvolution(50, model)
    # ne.load_population()
    ne.evolve(fitness=900, save_name="population.pkl", metrics=0, plot=True)
    
    model = ne.population.best_individual()
    env = gym.make(game, render_mode="human")
    state, info = env.reset()
    total_reward = 0
    while True:
        for individual in ne.population:
            while True:
                with torch.no_grad():
                    action = individual.model(ne.simclr.extract_features(torch.FloatTensor(state / 255).permute(2, 0, 1).unsqueeze(0).to(device))).cpu().squeeze(0).detach().numpy()
                    state, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    if terminated or truncated:
                        print(f"Last reward: {total_reward}")
                        total_reward = 0
                        state, info = env.reset()
                        break