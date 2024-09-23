import torch
from torch import nn
import torch.nn.functional as F
# import gymnasium as gym
# from PIL import Image

# env = gym.make("CarRacing-v2")
# layer = nn.MaxPool2d(2, 2)

# state, info = env.reset()
# for i in range(1000):
#     # torch_state = torch.from_numpy(state).permute(2, 0, 1)
#     # for _ in range(4):
#     #     torch_state = layer(torch_state)
#     #     print(torch_state.shape)
#     # state = torch_state.permute(1, 2, 0).numpy()
#     image = Image.fromarray(state).resize((48, 48)).convert('L')
#     image.save(f"./image/{i}.jpg")
#     state, reward, terminated, truncated, info = env.step(env.action_space.sample())
    
def calculate_discrete_similarity(old_actions, new_actions):
    """Calculate the percentage of matching actions between old and new models."""
    matches = (old_actions == new_actions).sum()
    total_actions = len(old_actions)
    similarity_percentage = matches / total_actions
    return similarity_percentage

# Example usage:
old_actions = torch.tensor([0, 1, 2, 1, 3])  # Actions from old model
new_actions = torch.tensor([0, 1, 2, 0, 3])  # Actions from new model

similarity = calculate_discrete_similarity(old_actions, new_actions)
print(f"Discrete Action Similarity: {similarity * 100:.2f}%")


def calculate_continuous_similarity(old_actions, new_actions, threshold=0.1):
    """Calculate the percentage of actions where the difference is below the threshold."""
    distances = torch.norm(old_actions - new_actions, dim=-1)  # L2 norm (Euclidean distance)
    matches = (distances <= threshold).sum()
    total_actions = len(old_actions)
    similarity_percentage = matches / total_actions
    return similarity_percentage

# Example usage:
old_actions = torch.tensor([[0.5, -0.2], [0.9, 0.1], [-0.4, 0.5]])  # Actions from old model
new_actions = torch.tensor([[0.52, -0.21], [0.87, 0.12], [-0.5, 0.52]])  # Actions from new model

similarity = calculate_continuous_similarity(old_actions, new_actions, threshold=0.05)
print(f"Continuous Action Similarity: {similarity * 100:.2f}%")


def calculate_cosine_similarity(old_actions, new_actions):
    """Calculate the average cosine similarity between old and new model actions."""
    cos = nn.CosineSimilarity(dim=-1)
    similarities = cos(old_actions, new_actions)
    average_similarity = similarities.mean()
    return average_similarity

# Example usage:
old_actions = torch.tensor([[0.5, -0.2], [0.9, 0.1], [-0.4, 0.5]])  # Actions from old model
new_actions = torch.tensor([[0.52, -0.21], [0.87, 0.12], [-0.5, 0.52]])  # Actions from new model

cosine_similarity = calculate_cosine_similarity(old_actions, new_actions)
print(f"Average Cosine Similarity: {cosine_similarity:.2f}")


import torch

def normalize_action(action, min_val, max_val):
    """Normalize action based on its range [min_val, max_val]"""
    if torch.isinf(min_val) or torch.isinf(max_val):
        # Use tanh for unbounded ranges (assuming symmetrical range [-inf, inf])
        normalized = torch.tanh(action)
    else:
        # Min-max normalization for bounded ranges
        normalized = (action - min_val) / (max_val - min_val)
    return normalized

def normalize_action_tensor(action_tensor, min_values, max_values):
    """Normalize each element in the action tensor based on its specific range."""
    normalized_actions = torch.empty_like(action_tensor)
    for i, (action, min_val, max_val) in enumerate(zip(action_tensor, min_values, max_values)):
        normalized_actions[i] = normalize_action(action, min_val, max_val)
    return normalized_actions

# Example action tensor with mixed ranges
action_tensor = torch.tensor([0.5, -0.8, 100, 2.0])  # Example actions
min_values = torch.tensor([0.0, -1.0, -float('inf'), -3.5])  # Corresponding min values
max_values = torch.tensor([1.0, 1.0, float('inf'), 3.5])  # Corresponding max values

# Normalize actions
normalized_actions = normalize_action_tensor(action_tensor, min_values, max_values)
print(normalized_actions)



def validate(batch1, batch2, threshold=0.5, method='zscore', scale_range=(-1, 1)):
    combined_batches = torch.cat([batch1, batch2], dim=0)
    
    if method == 'minmax':
        min_val = combined_batches.min(dim=0, keepdim=True)[0]
        max_val = combined_batches.max(dim=0, keepdim=True)[0]
        normalized_batch1 = scale_range[0] + (batch1 - min_val) * (scale_range[1] - scale_range[0]) / (max_val - min_val)
        normalized_batch2 = scale_range[0] + (batch2 - min_val) * (scale_range[1] - scale_range[0]) / (max_val - min_val)
    elif method == 'zscore':
        mean = combined_batches.mean(dim=0, keepdim=True)
        std = combined_batches.std(dim=0, keepdim=True)
        normalized_batch1 = (batch1 - mean) / std
        normalized_batch2 = (batch2 - mean) / std
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return F.pairwise_distance(normalized_batch1, normalized_batch2).mean().item() > threshold
    


# Example usage:
batch1 = torch.tensor([[0.5, -0.8, 100, 2.0],
                       [0.1, 0.5, -50, -2.5],
                       [0.9, 1.0, 10, 1.0],
                       [0.0, -0.5, 200, -3.5],
                       [0.7, 0.8, -100, 0.0]])

batch2 = torch.tensor([[0.4, -0.9, 120, 2.5],
                       [0.2, 0.4, -60, -2.0],
                       [0.85, 0.9, 15, 0.5],
                       [0.1, -0.6, 190, -3.0],
                       [0.75, 0.85, -90, -0.5]])


# Normalize both batches using combined min and max values
distances = validate(batch1, batch2)
print("Euclidean Distances:\n", distances)

print()
print()
print()

import torch
import torch.nn.functional as F

# Example action batches
batch1 = torch.tensor([[0.5, -0.8, 100, 2.0],
                       [0.1, 0.5, -50, -2.5]])

batch2 = torch.tensor([[0.4, -0.9, 120, 2.5],
                       [0.2, 0.4, -60, -2.0]])

# Normalize the data before distance computations (optional)
def normalize(batch1, batch2):
    combined = torch.cat([batch1, batch2], dim=0)
    min_val = combined.min(dim=0, keepdim=True)[0]
    max_val = combined.max(dim=0, keepdim=True)[0]
    return 2 * (batch1 - min_val) / (max_val - min_val) - 1, 2 * (batch2 - min_val) / (max_val - min_val) - 1

batch1_normalized, batch2_normalized = normalize(batch1, batch2)

# Euclidean Distance
euclidean_distance = F.pairwise_distance(batch1_normalized, batch2_normalized, p=2)
print("Euclidean Distance:\n", euclidean_distance)

# Cosine Similarity
cosine_similarity = (F.cosine_similarity(batch1_normalized, batch2_normalized, dim=1) ** 2).sum()**0.5
print("Cosine Similarity:\n", cosine_similarity)

# Example action sequences
old_actions = [0, 1, 2, 3, 2]
new_actions = [0, 1, 2, 2, 2]

# Calculate Hamming Distance
def calculate_hamming(seq1, seq2):
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be of the same length")
    return sum(el1 != el2 for el1, el2 in zip(seq1, seq2))

hamming_distance = calculate_hamming(old_actions, new_actions)
print("Hamming Distance:", hamming_distance)