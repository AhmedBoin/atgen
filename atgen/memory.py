from collections import deque
from typing import List
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

"""
Parameters for FailureReplayBuffer:
- Buffer Size: The maximum number of failure transitions to store in the buffer.
- Steps: How many steps (state-action pair) to store for each failure before failure occur.
- Signal: (method) failure signal to store the current buffer
- Similarity Threshold: if dis-similar threshold pass apply the model
- Failure Type: if model fail in any case, or the total sum similarity for all cases
- Priority: depending on Worst failure (boolean)

Input:
- state: Tensor[1, c, h, w]
- action: Tensor[1, a]
- reward: int
"""

class FailureContainer:
    def __init__(self, state, action, reward=None):
        self.state: List[torch.Tensor] = state
        self.action: List[torch.Tensor] = action
        self.reward: List[int] = reward


def hamming_distance(old_actions: torch.Tensor, new_actions: torch.Tensor) -> float:
    """Calculate the percentage of matching actions between old and new models."""
    return 1 - (old_actions == new_actions).mean().item()


def euclidean_distances(batch1: torch.Tensor, batch2: torch.Tensor, method='zscore', scale_range=(-1, 1)) -> float:
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
        raise ValueError(f"Unknown normalizing method: {method}")
    
    return F.pairwise_distance(normalized_batch1, normalized_batch2).mean().item()


class FailureReplayBuffer:
    def __init__(self, buffer_size: int, steps: int = 5, discrete_action: bool = False, 
                 threshold: float = 0.3, method='zscore', scale_range=(-1, 1), 
                 total_failure: bool = False, prioritize: bool = False):
        
        self.buffer_size = buffer_size
        self.buffer: List[FailureContainer] = deque(maxlen=buffer_size)
        self.state: List[torch.Tensor] = deque(maxlen=steps)
        self.action: List[torch.Tensor] = deque(maxlen=steps)
        self.reward: List[int] = deque(maxlen=steps)
        self.discrete_action = discrete_action
        self.threshold = threshold
        self.method = method
        self.scale_range = scale_range
        self.total_failure = total_failure
        self.prioritize = prioritize

    def step(self, state, action, reward=None):
        self.state.append(state)
        self.action.append(action)
        if self.prioritize:
            self.reward.append(reward)

    def signal(self):
        if self.prioritize:
            if len(self.buffer) == self.buffer_size:
                reward, idx = sum(self.reward), None
                for i, failure in enumerate(self.buffer):
                    failure: FailureContainer
                    if sum(failure.reward) > reward:
                        reward, idx = sum(failure.reward), i
                if idx is not None:
                    self.buffer.pop(idx)
                    self.buffer.append(FailureContainer(self.state, self.action, self.reward))
            else:
                self.buffer.append(FailureContainer(self.state, self.action, self.reward))
        else:
            self.buffer.append(FailureContainer(self.state, self.action))

    @torch.no_grad()
    def validate(self, model: nn.Sequential) -> bool:
        device = model.parameters().__next__().device
        if self.total_failure:
            states = torch.concat([failure.state for failure in self.buffer]).to(device)
            actions = torch.concat([failure.action for failure in self.buffer]).to(device)
            new_action: torch.Tensor = model(states)
            distance = hamming_distance(actions, new_action.argmax(1)) if self.discrete_action \
                else euclidean_distances(actions, new_action, self.method, self.scale_range)
            return distance > self.threshold
        else:
            for failure in self.buffer:
                new_action: torch.Tensor = model(failure.state)
                distance = hamming_distance(actions, new_action.argmax(1)) if self.discrete_action \
                    else euclidean_distances(actions, new_action, self.method, self.scale_range)
                if distance > self.threshold:
                    return False
            return True
        
    def clear(self):
        self.buffer.clear()
        self.state.clear()
        self.action.clear()
        self.reward.clear()

    def __len__(self) -> int:
        return len(self.buffer)
    
    