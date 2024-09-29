"""
The Revival of Natural Selection
"""
import pickle
from typing import Dict, Generic, List, Tuple, TypeVar, Union
from collections import deque
from itertools import chain
from typing import Deque
import torch
from torch import nn
import torch.nn.functional as F
import math


class Action:
    Good = "good"
    Bad = "bad"
    Normal = "normal"


class ContainerBuffer:
    def __init__(self, state, action, reward, action_type=Action.Normal, steps=0):
        self.state: torch.Tensor = torch.concat(list(state)[:steps], dim=0)
        self.action: torch.Tensor = torch.stack(list(action)[:steps], dim=0)
        self.reward: int = reward
        self.action_type: str = action_type

    def __str__(self) -> str:
        return f"Container(state: {self.state.shape}, action: {self.action.shape}, reward: {self.reward}, type: {self.action_type})"

    def __repr__(self) -> str:
        return self.__str__()
    

class CustomDeque(deque, Generic[TypeVar('T')]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def state(self):
        return torch.cat([item.state for item in self], dim=0)

    @property
    def action(self):
        return torch.cat([item.action for item in self], dim=0)
    
    @property
    def min(self):
        return min(self, key=lambda x: x.reward).reward if len(self) > 0 else 0
    
    @property
    def max(self):
        return max(self, key=lambda x: x.reward).reward if len(self) > 0 else 0
    
    def __str__(self) -> str:
        return f"{[i for i in self]}"
    
    def __len__(self) -> int:
        return super().__len__()


def discrete_similarity(old_actions: torch.Tensor, new_actions: torch.Tensor, actions: int) -> float:
    """Calculate similarity of discrete actions"""
    # Convert to one-hot vectors
    old_actions = F.one_hot(old_actions, num_classes=actions).float()
    new_actions = F.one_hot(new_actions, num_classes=actions).float()

    # Exponential Moving Average (EMA) smoothing
    smoothing_factor = 1/actions
    old_smoothed = torch.zeros_like(old_actions)
    old_smoothed[0] = old_actions[0]
    for i in range(1, len(old_actions)):
        old_smoothed[i] = smoothing_factor * old_actions[i] + (1 - smoothing_factor) * old_smoothed[i - 1]
    new_smoothed = torch.zeros_like(new_actions)
    new_smoothed[0] = new_actions[0]
    for i in range(1, len(new_actions)):
        new_smoothed[i] = smoothing_factor * new_actions[i] + (1 - smoothing_factor) * new_smoothed[i - 1]
    
    # Calculate similarity between two sequences
    return continues_similarity(old_smoothed, new_smoothed)


def continues_similarity(batch1: torch.Tensor, batch2: torch.Tensor) -> float:
    """Calculate similarity of Continues actions"""
    return F.relu(F.cosine_similarity(batch1, batch2)).mean().item()


class ReplayBuffer:
    def __init__(self, buffer_size: int = 0, steps: int = 0, dilation: int = 0, discrete_action: bool = False, 
                 similarity_threshold: float = 1.0, similarity_cohort: int = 1, accumulative_reward=False, 
                 gamma=0.95, reward_range=(0, 0), prioritize: bool = True, patience=10):
        
        self.steps: int = steps
        self.dilation: int = dilation
        self.good_buffer: CustomDeque[ContainerBuffer] = CustomDeque(maxlen=buffer_size)
        self.bad_buffer: CustomDeque[ContainerBuffer] = CustomDeque(maxlen=buffer_size)
        self.state: List[torch.Tensor] = []
        self.action: List[torch.Tensor] = []
        self.reward: List[int] = []

        self.discrete_action: bool = discrete_action
        self.similarity_threshold: float = similarity_threshold
        self.similarity_cohort: int = similarity_cohort
        self.offsprings: List[Tuple[float, nn.Sequential]] = []

        self.upper_bound = max(reward_range)
        self.lower_bound = min(reward_range)
        self.currant_action = Action.Normal

        self.gamma = gamma
        self.accumulative_reward = accumulative_reward
        self.prioritize = prioritize
        self.patience = patience

    def is_available(self) -> bool:
        return self.good_buffer.maxlen > 0 and self.steps > 0

    @property
    def size(self) -> int:
        return self.good_buffer.maxlen


    def step(self, state, action, reward):
        self.state.append(state)
        self.action.append(action)
        self.reward.append(reward)


    def signal(self, action_type: str, reward: int):
        new_container = ContainerBuffer(self.state, self.action, reward, action_type, self.steps)
        
        if self.prioritize:
            if action_type == Action.Good:
                if len(self.good_buffer) < self.good_buffer.maxlen:
                    self.good_buffer.append(new_container)
                else:
                    # Sort the good_buffer by reward (ascending) and replace the lowest reward if new reward is higher
                    sorted_good = sorted(self.good_buffer, key=lambda c: c.reward)
                    if reward > sorted_good[0].reward:
                        self.good_buffer.remove(sorted_good[0])
                        self.good_buffer.append(new_container)
                        
            else:  # Action.Bad
                if len(self.bad_buffer) < self.bad_buffer.maxlen:
                    self.bad_buffer.append(new_container)
                else:
                    # Sort the bad_buffer by reward (descending) and replace the highest reward if new reward is lower
                    sorted_bad = sorted(self.bad_buffer, key=lambda c: c.reward, reverse=True)
                    if reward < sorted_bad[0].reward:
                        self.bad_buffer.remove(sorted_bad[0])
                        self.bad_buffer.append(new_container)
        else:
            if action_type == Action.Good:
                self.good_buffer.append(new_container)
            else:
                self.bad_buffer.append(new_container)

    def track(self):
        # G(t) = g(t) + gamma * g(t+1)
        current_reward = self.reward[-1]
        modified_reward = [current_reward]
        for reward in reversed(self.reward[:-1]):
            current_reward += self.gamma * reward
            modified_reward.append(current_reward)
        self.reward = list(reversed(modified_reward))

        for reward in self.reward:
            if reward >= self.upper_bound:
                if self.currant_action != Action.Good: 
                    self.upper_bound = reward
                    self.signal(Action.Good, reward)
                self.currant_action = Action.Good
            elif reward <= self.lower_bound:
                if self.currant_action != Action.Bad:
                    self.lower_bound = reward
                    self.signal(Action.Bad, reward)
                self.currant_action = Action.Bad
            else:
                self.currant_action = Action.Normal

        # clean episode
        self.state.clear()
        self.action.clear()
        self.reward.clear()
        self.currant_action = Action.Normal

        if self.prioritize:
            self.half_clear()


    def _validate(self, model: nn.Sequential) -> float:
        g_similarity, b_similarity = 0, 0
        if len(self.good_buffer) > 0:
            good_new_action: torch.Tensor = model(self.good_buffer.state)
            if self.discrete_action:
                g_similarity = discrete_similarity(self.good_buffer.action, good_new_action.argmax(1), good_new_action.shape[1]) * len(self.good_buffer)
            else:
                g_similarity = continues_similarity(self.good_buffer.action, good_new_action) * len(self.good_buffer)
        if len(self.bad_buffer) > 0:
            bad_new_action: torch.Tensor = model(self.bad_buffer.state)
            if self.discrete_action:
                b_similarity = (1-discrete_similarity(self.bad_buffer.action, bad_new_action.argmax(1), bad_new_action.shape[1])) * len(self.bad_buffer)
            else:
                b_similarity = (1-continues_similarity(self.bad_buffer.action, bad_new_action)) * len(self.bad_buffer)
        if len(self) > 0:
            similarity = (g_similarity + b_similarity) / len(self)
            return similarity
        else:
            return 1.0
        

    @torch.no_grad()
    def validate(self, model: nn.Sequential) -> nn.Sequential:
        if len(self.offsprings) < self.similarity_cohort:
            distance = self._validate(model)
            if distance >= self.similarity_threshold:
                self.offsprings.clear()
                return model
            self.offsprings.append((distance, model))
        else:
            self.offsprings.sort(key=lambda x: x[0], reverse=True)
            model = self.offsprings[0][1]
            self.offsprings.clear()
            return model
            

    def clear(self):
        self.good_buffer.clear()
        self.bad_buffer.clear()
        self.state.clear()
        self.action.clear()
        self.reward.clear()
        self.upper_bound = 0
        self.lower_bound = 0
        self.currant_action = Action.Normal

    
    def half_clear(self):
        # Remove the top 50% from bad_buffer (worst half based on reward)
        if len(self.bad_buffer) == self.bad_buffer.maxlen:
            sorted_bad = sorted(self.bad_buffer, key=lambda x: x.reward, reverse=True)  # Higher rewards are worse
            half_len = len(sorted_bad) // 2
            self.bad_buffer = CustomDeque(sorted_bad[half_len:], maxlen=self.bad_buffer.maxlen)
        
        # Remove the lowest 50% from good_buffer (best half based on reward)
        if len(self.good_buffer) == self.good_buffer.maxlen:
            sorted_good = sorted(self.good_buffer, key=lambda x: x.reward)  # Lower rewards are better
            half_len = len(sorted_good) // 2
            self.good_buffer = CustomDeque(sorted_good[half_len:], maxlen=self.good_buffer.maxlen)


    def __len__(self) -> int:
        return len(self.good_buffer) + len(self.bad_buffer)
    
    def save(self, file_name: str="memory.pkl"):
        with open(f'{file_name}', 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, file_name: str="memory.pkl") -> "ReplayBuffer":
        with open(file_name, 'rb') as file:
            return pickle.load(file)
    

