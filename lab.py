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
    
# def calculate_discrete_similarity(old_actions, new_actions):
#     """Calculate the percentage of matching actions between old and new models."""
#     matches = (old_actions == new_actions).sum()
#     total_actions = len(old_actions)
#     similarity_percentage = matches / total_actions
#     return similarity_percentage

# # Example usage:
# old_actions = torch.tensor([0, 1, 2, 1, 3])  # Actions from old model
# new_actions = torch.tensor([0, 1, 2, 0, 3])  # Actions from new model

# similarity = calculate_discrete_similarity(old_actions, new_actions)
# print(f"Discrete Action Similarity: {similarity * 100:.2f}%")


# def calculate_continuous_similarity(old_actions, new_actions, threshold=0.1):
#     """Calculate the percentage of actions where the difference is below the threshold."""
#     distances = torch.norm(old_actions - new_actions, dim=-1)  # L2 norm (Euclidean distance)
#     matches = (distances <= threshold).sum()
#     total_actions = len(old_actions)
#     similarity_percentage = matches / total_actions
#     return similarity_percentage

# # Example usage:
# old_actions = torch.tensor([[0.5, -0.2], [0.9, 0.1], [-0.4, 0.5]])  # Actions from old model
# new_actions = torch.tensor([[0.52, -0.21], [0.87, 0.12], [-0.5, 0.52]])  # Actions from new model

# similarity = calculate_continuous_similarity(old_actions, new_actions, threshold=0.05)
# print(f"Continuous Action Similarity: {similarity * 100:.2f}%")


# def calculate_cosine_similarity(old_actions, new_actions):
#     """Calculate the average cosine similarity between old and new model actions."""
#     cos = nn.CosineSimilarity(dim=-1)
#     similarities = cos(old_actions, new_actions)
#     average_similarity = similarities.mean()
#     return average_similarity

# # Example usage:
# old_actions = torch.tensor([[0.5, -0.2], [0.9, 0.1], [-0.4, 0.5]])  # Actions from old model
# new_actions = torch.tensor([[0.52, -0.21], [0.87, 0.12], [-0.5, 0.52]])  # Actions from new model

# cosine_similarity = calculate_cosine_similarity(old_actions, new_actions)
# print(f"Average Cosine Similarity: {cosine_similarity:.2f}")


# import torch

# def normalize_action(action, min_val, max_val):
#     """Normalize action based on its range [min_val, max_val]"""
#     if torch.isinf(min_val) or torch.isinf(max_val):
#         # Use tanh for unbounded ranges (assuming symmetrical range [-inf, inf])
#         normalized = torch.tanh(action)
#     else:
#         # Min-max normalization for bounded ranges
#         normalized = (action - min_val) / (max_val - min_val)
#     return normalized

# def normalize_action_tensor(action_tensor, min_values, max_values):
#     """Normalize each element in the action tensor based on its specific range."""
#     normalized_actions = torch.empty_like(action_tensor)
#     for i, (action, min_val, max_val) in enumerate(zip(action_tensor, min_values, max_values)):
#         normalized_actions[i] = normalize_action(action, min_val, max_val)
#     return normalized_actions

# # Example action tensor with mixed ranges
# action_tensor = torch.tensor([0.5, -0.8, 100, 2.0])  # Example actions
# min_values = torch.tensor([0.0, -1.0, -float('inf'), -3.5])  # Corresponding min values
# max_values = torch.tensor([1.0, 1.0, float('inf'), 3.5])  # Corresponding max values

# # Normalize actions
# normalized_actions = normalize_action_tensor(action_tensor, min_values, max_values)
# print(normalized_actions)



# def validate(batch1, batch2, threshold=0.5, method='zscore', scale_range=(-1, 1)):
#     combined_batches = torch.cat([batch1, batch2], dim=0)
    
#     if method == 'minmax':
#         min_val = combined_batches.min(dim=0, keepdim=True)[0]
#         max_val = combined_batches.max(dim=0, keepdim=True)[0]
#         normalized_batch1 = scale_range[0] + (batch1 - min_val) * (scale_range[1] - scale_range[0]) / (max_val - min_val)
#         normalized_batch2 = scale_range[0] + (batch2 - min_val) * (scale_range[1] - scale_range[0]) / (max_val - min_val)
#     elif method == 'zscore':
#         mean = combined_batches.mean(dim=0, keepdim=True)
#         std = combined_batches.std(dim=0, keepdim=True)
#         normalized_batch1 = (batch1 - mean) / std
#         normalized_batch2 = (batch2 - mean) / std
#     else:
#         raise ValueError(f"Unknown method: {method}")
    
#     return F.pairwise_distance(normalized_batch1, normalized_batch2).mean().item() > threshold
    


# # Example usage:
# batch1 = torch.tensor([[0.5, -0.8, 100, 2.0],
#                        [0.1, 0.5, -50, -2.5],
#                        [0.9, 1.0, 10, 1.0],
#                        [0.0, -0.5, 200, -3.5],
#                        [0.7, 0.8, -100, 0.0]])

# batch2 = torch.tensor([[0.4, -0.9, 120, 2.5],
#                        [0.2, 0.4, -60, -2.0],
#                        [0.85, 0.9, 15, 0.5],
#                        [0.1, -0.6, 190, -3.0],
#                        [0.75, 0.85, -90, -0.5]])


# # Normalize both batches using combined min and max values
# distances = validate(batch1, batch2)
# print("Euclidean Distances:\n", distances)

# print()
# print()
# print()

# import torch
# import torch.nn.functional as F

# # Example action batches
# batch1 = torch.tensor([[0.5, -0.8, 100, 2.0],
#                        [0.1, 0.5, -50, -2.5]])

# batch2 = torch.tensor([[0.4, -0.9, 120, 2.5],
#                        [0.2, 0.4, -60, -2.0]])

# # Normalize the data before distance computations (optional)
# def normalize(batch1, batch2):
#     combined = torch.cat([batch1, batch2], dim=0)
#     min_val = combined.min(dim=0, keepdim=True)[0]
#     max_val = combined.max(dim=0, keepdim=True)[0]
#     return 2 * (batch1 - min_val) / (max_val - min_val) - 1, 2 * (batch2 - min_val) / (max_val - min_val) - 1

# batch1_normalized, batch2_normalized = normalize(batch1, batch2)

# # Euclidean Distance
# euclidean_distance = F.pairwise_distance(batch1_normalized, batch2_normalized, p=2)
# print("Euclidean Distance:\n", euclidean_distance)

# # Cosine Similarity
# cosine_similarity = (F.cosine_similarity(batch1_normalized, batch2_normalized, dim=1) ** 2).sum()**0.5
# print("Cosine Similarity:\n", cosine_similarity)

# # Example action sequences
# old_actions = [0, 1, 2, 3, 2]
# new_actions = [0, 1, 2, 2, 2]

# # Calculate Hamming Distance
# def calculate_hamming(seq1, seq2):
#     if len(seq1) != len(seq2):
#         raise ValueError("Sequences must be of the same length")
#     return sum(el1 != el2 for el1, el2 in zip(seq1, seq2))

# hamming_distance = calculate_hamming(old_actions, new_actions)
# print("Hamming Distance:", hamming_distance)


# import numpy as np

# def detect_sudden_change(rewards, window_size=5, threshold=1.0):
#     # Compute the moving average of the rewards
#     moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    
#     sudden_changes = []
#     for i in range(len(moving_avg)):
#         if abs(rewards[i + window_size - 1] - moving_avg[i]) > threshold:
#             sudden_changes.append(i + window_size - 1)
    
#     return sudden_changes

# # Example reward series
# rewards = [1, 2, 2.5, 3, 3.1, 10, 3.2, 3.1, 3.0, 0] # custom game
# rewards = [1, 1, 2, 1, 1, -1, -1, -1, -1, -1, 1, 1, 2] # CarRacing
# rewards = [-0.3, -1, -0.3, -0,3, 0, -1, -0,3, 0, 200] # LunarLander

# # Detect sudden changes with a threshold of 1.0
# changes = detect_sudden_change(rewards, window_size=3, threshold=1.0)
# print("Sudden changes detected at indices:", changes)

# def z_score_anomaly_detection(rewards, threshold=2.0):
#     mean = np.mean(rewards)
#     std_dev = np.std(rewards)
    
#     z_scores = [(r - mean) / std_dev for r in rewards]
#     anomalies = [i for i, z in enumerate(z_scores) if abs(z) > threshold]
    
#     return anomalies

# # Detect anomalies using Z-score method
# anomalies = z_score_anomaly_detection(rewards, threshold=2.0)
# print("Anomalies detected at indices:", anomalies)

# def cusum(rewards, threshold=5):
#     mean_reward = np.mean(rewards)
#     pos_cumsum = np.zeros(len(rewards))
#     neg_cumsum = np.zeros(len(rewards))
    
#     sudden_changes = []
    
#     for i in range(1, len(rewards)):
#         pos_cumsum[i] = max(0, pos_cumsum[i-1] + rewards[i] - mean_reward)
#         neg_cumsum[i] = max(0, neg_cumsum[i-1] - (rewards[i] - mean_reward))
        
#         if pos_cumsum[i] > threshold or neg_cumsum[i] > threshold:
#             sudden_changes.append(i)
#             pos_cumsum[i] = neg_cumsum[i] = 0  # Reset after detection
    
#     return sudden_changes

# # Detect sudden changes using CUSUM
# changes = cusum(rewards, threshold=2.0)
# print("Sudden changes detected at indices:", changes)

# def ewma_anomaly_detection(rewards, span=3, threshold=1.5):
#     ewma = pd.Series(rewards).ewm(span=span).mean()
#     anomalies = [i for i in range(len(rewards)) if abs(rewards[i] - ewma[i]) > threshold]
#     return anomalies

# # Detect anomalies using EWMA
# import pandas as pd
# anomalies = ewma_anomaly_detection(rewards, span=3, threshold=1.5)
# print("Anomalies detected at indices:", anomalies)

###########################################################################################################################
# import numpy as np

# def apply_discount(rewards, gamma):
#     """
#     Apply discounting factor to the reward sequence.
    
#     :param rewards: List of rewards [r1, r2, ..., rN]
#     :param gamma: Discount factor, typically between 0 and 1.
#     :return: Discounted rewards.
#     """
#     discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
#     running_add = 0
#     for t in reversed(range(len(rewards))):
#         running_add = rewards[t] + gamma * running_add
#         discounted_rewards[t] = running_add
#     return discounted_rewards

# def detect_sudden_change(rewards, window_size=3, threshold=1.0):
#     """
#     Detect sudden changes in a reward sequence using moving average with thresholding.
    
#     :param rewards: Sequence of rewards (could be discounted or original rewards).
#     :param window_size: Window size for the moving average.
#     :param threshold: Threshold for detecting sudden change.
#     :return: List of indices where sudden changes are detected.
#     """
#     moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
#     sudden_changes = []
#     for i in range(len(moving_avg)):
#         if abs(rewards[i + window_size - 1] - moving_avg[i]) > threshold:
#             sudden_changes.append(i + window_size - 1)
    
#     return sudden_changes

# # Example rewards from LunarLander, CarRacing, etc.
# rewards = [-0.3, -1, -0.3, -0.3, 0, -1, -0.3, 0, 200]  # LunarLander
# gamma = 0.9  # Example discount factor

# # Step 1: Apply discount factor
# discounted_rewards = apply_discount(rewards, gamma)
# print("Discounted Rewards:", discounted_rewards)

# # Step 2: Detect sudden changes
# sudden_changes = detect_sudden_change(discounted_rewards, window_size=3, threshold=1.0)
# print("Sudden changes detected at indices:", sudden_changes)



# # Example reward sequences from different games
# custom_game_rewards = [1, 2, 2.5, 3, 3.1, 10, 3.2, 3.1, 3.0, 0]  # Custom game
# carracing_rewards = [1, 1, 2, 1, 1, -1, -1, -1, -1, -1, 1, 1, 2]  # CarRacing
# lunarlander_rewards = [-0.3, -1, -0.3, -0.3, 0, -1, -0.3, 0, 200]  # LunarLander

# # Example discount factor
# gamma = 0.9  # Discount factor for smoothing

# # Apply discount factor to each reward sequence
# discounted_custom_game_rewards = apply_discount(custom_game_rewards, gamma)
# discounted_carracing_rewards = apply_discount(carracing_rewards, gamma)
# discounted_lunarlander_rewards = apply_discount(lunarlander_rewards, gamma)

# # Detect sudden changes with the same function (adjust threshold for each game)
# sudden_changes_custom_game = detect_sudden_change(discounted_custom_game_rewards, window_size=3, threshold=2.0)
# sudden_changes_carracing = detect_sudden_change(discounted_carracing_rewards, window_size=3, threshold=2.0)
# sudden_changes_lunarlander = detect_sudden_change(discounted_lunarlander_rewards, window_size=3, threshold=10.0)

# print("Sudden changes in custom game rewards:", sudden_changes_custom_game)
# print("Sudden changes in CarRacing rewards:", sudden_changes_carracing)
# print("Sudden changes in LunarLander rewards:", sudden_changes_lunarlander)

# def detect_changes_for_game(rewards, gamma=0.9, window_size=3, threshold=1.0):
#     # Step 1: Apply discount
#     discounted_rewards = apply_discount(rewards, gamma)
    
#     # Step 2: Detect sudden changes
#     sudden_changes = detect_sudden_change(discounted_rewards, window_size, threshold)
    
#     return sudden_changes

# # Detect sudden changes for multiple games
# custom_game_changes = detect_changes_for_game(custom_game_rewards, gamma=0.9, window_size=3, threshold=2.0)
# carracing_changes = detect_changes_for_game(carracing_rewards, gamma=0.9, window_size=3, threshold=2.0)
# lunarlander_changes = detect_changes_for_game(lunarlander_rewards, gamma=0.9, window_size=3, threshold=10.0)

# print("Sudden changes for Custom Game:", custom_game_changes)
# print("Sudden changes for CarRacing:", carracing_changes)
# print("Sudden changes for LunarLander:", lunarlander_changes)



# ###########################################################################################################################
# import numpy as np

# def calculate_acceleration(rewards):
#     """
#     Calculate acceleration by computing the second-order difference of rewards.
#     :param rewards: List of rewards.
#     :return: Accelerations (second-order difference).
#     """
#     velocity = np.diff(rewards)  # First-order difference (velocity)
#     acceleration = np.diff(velocity)  # Second-order difference (acceleration)
#     return acceleration

# def detect_sudden_change_adaptive(rewards, accel_threshold=1.0, sign_change_threshold=1.0):
#     """
#     Detect sudden changes in reward sequence based on acceleration and sign change in rewards.
    
#     :param rewards: List of rewards.
#     :param accel_threshold: Threshold for detecting sudden acceleration changes.
#     :param sign_change_threshold: Threshold for detecting sign changes.
#     :return: List of indices where sudden changes are detected.
#     """
#     accelerations = calculate_acceleration(rewards)
#     sudden_changes = []

#     # Check acceleration
#     for i in range(1, len(accelerations)):
#         if abs(accelerations[i]) > accel_threshold:
#             sudden_changes.append(i + 1)  # Offset due to double diff
        
#         # Detect sign changes in rewards (positive to negative or negative to positive)
#         if np.sign(rewards[i + 1]) != np.sign(rewards[i]) and abs(rewards[i + 1] - rewards[i]) > sign_change_threshold:
#             sudden_changes.append(i + 1)

#     return sudden_changes

# # Example reward sequences from different games
# custom_game_rewards = [1, 2, 2.5, 3, 3.1, 10, 3.2, 3.1, 3.0, 0]  # Custom game
# carracing_rewards = [1, 1, 2, 1, 1, -1, -1, -1, -1, -1, 1, 1, 2]  # CarRacing
# lunarlander_rewards = [-0.3, -1, -0.3, -0.3, 0, -1, -0.3, 0, 200]  # LunarLander

# # Apply the adaptive detection method
# custom_game_changes = detect_sudden_change_adaptive(custom_game_rewards, accel_threshold=2.0, sign_change_threshold=1.0)
# carracing_changes = detect_sudden_change_adaptive(carracing_rewards, accel_threshold=2.0, sign_change_threshold=1.0)
# lunarlander_changes = detect_sudden_change_adaptive(lunarlander_rewards, accel_threshold=100.0, sign_change_threshold=100.0)

# print()
# print((np.diff(custom_game_rewards)))
# print((np.diff(carracing_rewards)))
# print((np.diff(lunarlander_rewards)))
# print(np.diff(np.diff(custom_game_rewards)))
# print(np.diff(np.diff(carracing_rewards)))
# print(np.diff(np.diff(lunarlander_rewards)))
# print("Sudden changes for Custom Game:", custom_game_changes)
# print("Sudden changes for CarRacing:", carracing_changes)
# print("Sudden changes for LunarLander:", lunarlander_changes)

# ###########################################################################################################################
# import numpy as np
# import matplotlib.pyplot as plt

# def calculate_velocity(rewards):
#     """
#     Calculate velocity (first-order difference of rewards).
#     :param rewards: List of rewards.
#     :return: Velocities (first-order difference).
#     """
#     return np.abs(np.diff(rewards))

# def calculate_acceleration(rewards):
#     """
#     Calculate acceleration (second-order difference of rewards).
#     :param rewards: List of rewards.
#     :return: Accelerations (second-order difference).
#     """
#     velocity = np.abs(np.diff(rewards))
#     acceleration = np.abs(np.diff(velocity))
#     return acceleration

# def plot_rewards_velocity_acceleration(rewards, game_name):
#     """
#     Plot rewards, velocity, and acceleration for a given game.
#     :param rewards: List of rewards for the game.
#     :param game_name: Name of the game.
#     """
#     # Calculate velocity and acceleration
#     velocity = apply_discount(calculate_velocity(rewards), gamma)
#     acceleration = apply_discount(calculate_acceleration(rewards), gamma)
    
#     # Plot rewards
#     plt.plot(list(range(0, len(rewards))), apply_discount(rewards, gamma), marker='o', color='b', label='Rewards')
    
#     # Plot velocity
#     plt.plot(list(range(1, len(rewards))), velocity, marker='o', color='g', label='Velocity')
    
#     # Plot acceleration
#     plt.plot(list(range(2, len(rewards))), acceleration, marker='o', color='r', label='Acceleration')
    
#     # Add grid and layout adjustment
#     plt.grid(True)
#     plt.legend()
#     plt.show()

# gamma = 0.0
# # Example reward sequences from different games
# custom_game_rewards = [1, 2, 2.5, 3, 3.1, 10, 3.2, 3.1, 3.0, 0]  # Custom game 5, 6, 9
# carracing_rewards = [1, 1, 2, 1, 1, -1, -1, -2, -1, -1, 1, 1, 2]  # CarRacing 5, 10
# lunarlander_rewards = [-0.3, -1, -0.3, -0.3, 0, -1, -0.3, 0, 200]  # LunarLander -1

# # Plot for Custom Game
# plot_rewards_velocity_acceleration(custom_game_rewards, "Custom Game")

# # Plot for CarRacing
# plot_rewards_velocity_acceleration(carracing_rewards, "CarRacing")

# # Plot for LunarLander
# plot_rewards_velocity_acceleration(lunarlander_rewards, "LunarLander")
# # Plot for Custom Game

###########################################################################################################################

# class RewardTracker:
#     def __init__(self, upper_bound, lower_bound):
#         """
#         Initialize the reward tracker with upper and lower bounds.
#         :param upper_bound: Upper bound for good actions.
#         :param lower_bound: Lower bound for bad actions.
#         """
#         self.upper_bound = upper_bound
#         self.lower_bound = lower_bound
#         self.state = "normal"  # Initial state is "normal"
    
#     def track_rewards(self, rewards):
#         """
#         Track rewards and detect transitions to good/bad actions based on bounds.
#         :param rewards: List of rewards.
#         :return: List of (index, status) where status is "good" or "bad".
#         """
#         transitions = []
        
#         for i, reward in enumerate(rewards):
#             if self.state == "normal":
#                 if reward > self.upper_bound:
#                     transitions.append((i, "good"))
#                     self.state = "above"
#                 elif reward < self.lower_bound:
#                     transitions.append((i, "bad"))
#                     self.state = "below"
            
#             elif self.state == "above":
#                 if reward <= self.upper_bound:
#                     self.state = "normal"
            
#             elif self.state == "below":
#                 if reward >= self.lower_bound:
#                     self.state = "normal"
        
#         return transitions

# # Example rewards from different games
# custom_game_rewards = [1, 2, 2.5, 3, 3.1, 10, 3.2, 3.1, 3.0, 0]  # Custom game
# carracing_rewards = [1, 1, 2, 1, 1, -1, -1, -1, -1, -1, 1, 1, 2]  # CarRacing
# lunarlander_rewards = [-0.3, -1, -0.3, -0.3, 0, -1, -0.3, 0, 200]  # LunarLander

# # Define upper and lower bounds (these can vary depending on the game)
# upper_bound = 5
# lower_bound = 0.5

# # Initialize the tracker with bounds
# tracker = RewardTracker(upper_bound, lower_bound)

# # Track rewards for each game
# custom_game_transitions = tracker.track_rewards(custom_game_rewards)
# carracing_transitions = tracker.track_rewards(carracing_rewards)
# lunarlander_transitions = tracker.track_rewards(lunarlander_rewards)

# # Display results
# print("Custom Game Transitions:", custom_game_transitions)
# print("CarRacing Transitions:", carracing_transitions)
# print("LunarLander Transitions:", lunarlander_transitions)



###########################################################################################################################

# def track_rewards(rewards, upper_bound, lower_bound):
#     """
#     Track rewards and detect transitions to good/bad actions based on bounds.
#     Transitions happen immediately if rewards cross either bound.
    
#     :param rewards: List of rewards.
#     :param upper_bound: Upper bound for good actions.
#     :param lower_bound: Lower bound for bad actions.
#     :return: Two lists: one with indices of good actions and another with indices of bad actions.
#     """
#     good_actions = []
#     bad_actions = []
#     state = "normal"  # Start in the normal state
    
#     for i, reward in enumerate(rewards):
#         if reward > upper_bound:
#             if state != "above": good_actions.append(i)
#             state = "above"
#         elif reward < lower_bound:
#             if state != "below": bad_actions.append(i)
#             state = "below"
#         else:
#             state = "normal"
        
#     return good_actions, bad_actions


# # Example reward sequences from different games
# custom_game_rewards = [1, 2, 2.5, 3, 3.1, 10, 3.2, 3.1, 3.0, 0]  # Custom game
# carracing_rewards = [1, 1, 2, 1, 1, -1, -1, -1, -1, -1, 1, 1, 2]  # CarRacing
# lunarlander_rewards = [-0.3, -1, -0.3, -0.3, 0, -1, -0.3, 0, 200]  # LunarLander

# # Track rewards for each game
# custom_game_good, custom_game_bad = track_rewards(custom_game_rewards, upper_bound=5, lower_bound=0.5)
# carracing_good, carracing_bad = track_rewards(carracing_rewards, upper_bound=0.1, lower_bound=-0.1)
# lunarlander_good, lunarlander_bad = track_rewards(lunarlander_rewards, upper_bound=100, lower_bound=-5)

# # Display results
# print("Custom Game Good Actions:", custom_game_good)
# print("Custom Game Bad Actions:", custom_game_bad)

# print("CarRacing Good Actions:", carracing_good)
# print("CarRacing Bad Actions:", carracing_bad)

# print("LunarLander Good Actions:", lunarlander_good)
# print("LunarLander Bad Actions:", lunarlander_bad)



# import torch
# import torch.nn.functional as F

# # 1. Convert to one-hot vectors
# def convert_to_one_hot(action_series, num_actions=4):
#     return F.one_hot(torch.tensor(action_series), num_classes=num_actions).float()

# # 2. Apply Exponential Moving Average (EMA) smoothing
# def smooth_actions_ema(one_hot_actions, smoothing_factor=0.25):
#     smoothed = torch.zeros_like(one_hot_actions)
#     smoothed[0] = one_hot_actions[0]  # Initialize with the first action
#     for i in range(1, len(one_hot_actions)):
#         smoothed[i] = smoothing_factor * one_hot_actions[i] + (1 - smoothing_factor) * smoothed[i - 1]
#     return smoothed

# # 3. Calculate Euclidean distance between two sequences
# def euclidean_distance(seq1, seq2):
#     return torch.sqrt(torch.sum((seq1 - seq2) ** 2, dim=1)).mean().item()

# # Example usage:
# action_series1 = [1, 1, 2, 0, 0, 2, 3, 2, 2, 3, 1, 1, 0]
# action_series2 = [1, 2, 2, 0, 1, 2, 3, 2, 1, 3, 1, 0, 0]

# # Convert to one-hot
# one_hot1 = convert_to_one_hot(action_series1)
# one_hot2 = convert_to_one_hot(action_series2)
# print("One-hot vectors:")
# print(one_hot1)
# print(one_hot2)

# # Apply EMA smoothing
# smoothed1 = smooth_actions_ema(one_hot1, smoothing_factor=0.3)
# smoothed2 = smooth_actions_ema(one_hot2, smoothing_factor=0.3)
# print("\nSmoothed one-hot vectors:")
# print(smoothed1)
# print(smoothed2)

# # Calculate Euclidean distance
# distance = euclidean_distance(smoothed1, smoothed2)
# print(f"Euclidean Distance: {distance}")


# import torch

# def calculate_balanced_ratio(distances: torch.Tensor, labels: torch.Tensor, epsilon=1e-8):
#     """
#     Calculate a ratio that reflects how balanced positive and negative distances are.
    
#     :param distances: Tensor of shape (n,), where n is the number of vectors.
#                       Contains the Euclidean distances between corresponding vectors.
#     :param labels: Tensor of shape (n,), containing 1 for positive examples and 0 for negative examples.
#     :param epsilon: Small value to prevent division by zero.
    
#     :return: A ratio that reflects balance: near 1 if positive and negative distances are close,
#              near 0 if there's a significant difference.
#     """
#     # Separate distances for positive and negative examples
#     positive_distances = distances[labels == 1]
#     negative_distances = distances[labels == 0]
    
#     # Calculate mean distances
#     mean_positive_distance = positive_distances.mean().item()
#     mean_negative_distance = negative_distances.mean().item()
    
#     # Calculate the absolute difference and sum
#     abs_difference = abs(mean_positive_distance - mean_negative_distance)
#     total_distance = mean_positive_distance + mean_negative_distance
    
#     # Avoid division by zero by adding a small epsilon
#     if total_distance == 0:
#         return 1.0  # Both distances are zero, return max ratio (perfect balance)
    
#     # Calculate the ratio: closer to 1 if distances are balanced, closer to 0 if they are different
#     ratio = 1 - (abs_difference / (total_distance + epsilon))
    
#     return ratio

# # Example usage
# distances = torch.tensor([0.1, 0.2, 0.3, 1.5, 2.0] * 10)
# labels = torch.tensor([1, 0, 1, 0, 1] * 10)

# # Calculate the ratio
# balanced_ratio = calculate_balanced_ratio(distances, labels)
# print(f"Balanced ratio: {balanced_ratio:.4f}")