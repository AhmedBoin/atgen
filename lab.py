import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# CNN-based Variational Denoising Autoencoder
class CNN_VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(CNN_VAE, self).__init__()

        # Encoder (CNN for feature extraction)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True)
        )

        # Latent space: mean and log variance for VAE
        self.fc_mu = None# nn.Linear(64 * 2 * 2, latent_dim)
        self.fc_logvar = None# nn.Linear(64 * 2 * 2, latent_dim)

        # Decoder (Transposed Convolutions)
        self.fc_decode = None# nn.Linear(latent_dim, 64 * 2 * 2)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        if self.fc_mu is None:
            self.fc_mu = nn.Linear(x.shape[1], 20).to("mps")
            self.fc_logvar = nn.Linear(x.shape[1], 20).to("mps")
            self.fc_decode = nn.Linear(20, x.shape[1]).to("mps")
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std  # Reparametrization trick

    def decode(self, z):
        z = self.fc_decode(z)
        n = int(z.numel()/z.size(0)/64/2)
        z = z.view(z.size(0), 64, n, n)  # Reshape to feed into transposed convolutions
        return self.decoder(z)

    def forward(self, x):
        # Add noise to the input (Denoising Autoencoder)
        x_noisy = x + 0.2 * torch.randn_like(x)
        
        # Encode (extract features and produce latent variables)
        mu, logvar = self.encode(x_noisy)
        
        # Reparameterization trick to sample from latent space
        z = self.reparameterize(mu, logvar)
        
        # Decode (reconstruct image from latent space)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

# Loss function
def loss_function(recon_x, x, mu, logvar):
    # Reconstruction loss (Binary Cross Entropy)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL Divergence loss (for VAE regularization)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KLD

# Training loop
def train(model, train_loader, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data, _ in train_loader:
            data = data.to("mps")
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader.dataset)}')

# Load dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='../data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Initialize model, optimizer and train
model = CNN_VAE().to("mps")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train(model, train_loader, optimizer, epochs=10)