import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

USE_CUDA = True if torch.cuda.is_available() else False

class ConvLayer(nn.Module):
    def __init__(self, in_channels=3, out_channels=256, kernel_size=9):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=1)

    def forward(self, x):
        # Input: (batch_size, 3, 96, 96)
        # Output: (batch_size, 256, 88, 88)
        return F.relu(self.conv(x))

class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules=8, in_channels=256, out_channels=32, kernel_size=9, stride=2):
        super(PrimaryCaps, self).__init__()
        self.num_routes = num_capsules * 40 * 40  # Adjusted based on output size below
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
            for _ in range(num_capsules)])

    def forward(self, x):
        # Input: (batch_size, 256, 88, 88)
        u = [capsule(x) for capsule in self.capsules]  # u[i]: (batch_size, 32, 40, 40)
        u = torch.stack(u, dim=1)  # Stack all capsules, (batch_size, 8, 32, 40, 40)
        u = u.view(x.size(0), -1, 32)  # Flatten spatial dimensions, (batch_size, 12800, 32)
        return self.squash(u)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor  # Output: (batch_size, 12800, 32)

class DigitCaps(nn.Module):
    def __init__(self, num_capsules=10, num_routes=12800, in_channels=32, out_channels=1):
        super(DigitCaps, self).__init__()
        self.num_routes = num_routes
        self.in_channels = in_channels
        self.num_capsules = num_capsules
        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))

    def forward(self, x):
        # Input: (batch_size, 12800, 32)
        batch_size = x.size(0)
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)  # (batch_size, 12800, 10, 32, 1)
        W = torch.cat([self.W] * batch_size, dim=0)  # (batch_size, 12800, 10, 16, 32)
        u_hat = torch.matmul(W, x)  # (batch_size, 12800, 10, 16, 1)
        u_hat = u_hat.squeeze(4)  # (batch_size, 12800, 10, 16)

        b_ij = Variable(torch.zeros(1, self.num_routes, self.num_capsules, 1)).to(x.device)

        num_iterations = 3
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij, dim=1)  # Softmax over routes, (1, 12800, 10, 1)
            c_ij = torch.cat([c_ij] * batch_size, dim=0)  # (batch_size, 12800, 10, 1)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)  # Weighted sum, (batch_size, 1, 10, 16)
            v_j = self.squash(s_j)  # Apply squash function, 

            if iteration < num_iterations - 1: # (batch_size, 12800, 10, 16) . (batch_size, 12800, 10, 16)
                a_ij = torch.einsum("brch,brch->rc", u_hat, torch.cat([v_j] * self.num_routes, dim=1))  # Agreement (12800, 10)
                b_ij = b_ij + a_ij.unsqueeze(0).unsqueeze(3)  # Update coupling coefficients (1, 12800, 10, 1)
                # a_ij = torch.matmul(u_hat.transpose(2, 3), torch.cat([v_j] * self.num_routes, dim=1))  # Agreement (batch_size, 12800, 16, 16)
                # b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)  # Update coupling coefficients (1, 12800, 10, 1)

        return v_j.squeeze(1)  # Output: (batch_size, 10, 16)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor

class Decoder(nn.Module):
    def __init__(self, input_width=96, input_height=96, input_channel=3):
        super(Decoder, self).__init__()
        self.input_width = input_width
        self.input_height = input_height
        self.input_channel = input_channel
        self.reconstruction_layers = nn.Sequential(
            nn.Linear(10, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.input_height * self.input_width * self.input_channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Input: (batch_size, 10, 16)
        t = x.view(x.size(0), -1)  # Flatten to (batch_size, 160)
        reconstructions = self.reconstruction_layers(t)  # (batch_size, 96 * 96 * 3)
        reconstructions = reconstructions.view(-1, self.input_channel, self.input_height, self.input_width)  # Reshape back
        return reconstructions  # Output: (batch_size, 3, 96, 96)

class CapsNet(nn.Module):
    def __init__(self):
        super(CapsNet, self).__init__()
        self.conv_layer = ConvLayer()  # (batch_size, 3, 96, 96) -> (batch_size, 256, 88, 88)
        self.primary_capsules = PrimaryCaps()  # (batch_size, 256, 88, 88) -> (batch_size, 12800, 32)
        self.digit_capsules = DigitCaps()  # (batch_size, 12800, 32) -> (batch_size, 10, 16)
        self.decoder = Decoder()  # (batch_size, 10, 16) -> (batch_size, 3, 96, 96)

    def forward(self, x):
        x = self.conv_layer(x)  # (batch_size, 256, 88, 88)
        x = self.primary_capsules(x)  # (batch_size, 12800, 32)
        x = self.digit_capsules(x)  # (batch_size, 10, 16)
        reconstructions = self.decoder(x)  # (batch_size, 3, 96, 96)
        return x, reconstructions

    def reduce(self, x):
        x = self.conv_layer(x)  # (batch_size, 256, 88, 88)
        x = self.primary_capsules(x)  # (batch_size, 12800, 32)
        encoded = self.digit_capsules(x)  # (batch_size, 10, 16)
        return encoded.view(x.size(0), -1)  # Flatten to (batch_size, 160) for latent representation
    

class CapsuleLoss(nn.Module):
    def __init__(self, reconstruction_loss_weight=0.0005):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.reconstruction_loss_fn = nn.MSELoss()

    def forward(self, reconstructed_images, images):

        # Reconstruction loss
        reconstruction_loss = self.reconstruction_loss_fn(reconstructed_images, images)

        # Total loss
        total_loss = self.reconstruction_loss_weight * reconstruction_loss
        return total_loss
    
    
if __name__ == "__main__":
    # Initialize the model
    autoencoder = CapsNet()

    # Create a dummy input image (batch_size, channels, height, width)
    input_image = torch.randn(1, 3, 96, 96)

    # Pass the input through the autoencoder
    encoded, reconstructed = autoencoder(input_image)

    # Print the shapes to check correctness
    print(f"Input shape: {input_image.shape}")                 # Should be: (1, 3, 96, 96)
    print(f"Encoded shape: {encoded.shape}")                  # Should be: (1, 10, 1)
    print(f"Reconstructed shape: {reconstructed.shape}")      # Should be: (1, 3, 96, 96)

    # Test the reduce function
    latent_representation = autoencoder.reduce(input_image)
    print(f"Latent space shape: {latent_representation.shape}")  # Should be: (1, 10)