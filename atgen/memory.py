"""
The Revival of Natural Selection
"""

import pickle
from typing import Generic, List, Tuple, Deque, TypeVar
from collections import deque

import torch
from torch import nn
import torch.nn.functional as F


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
    
    # Calculate similarity between two sequences
    return ((F.cosine_similarity(old_actions, new_actions)+1)/2).mean().item()


def continues_similarity(old_actions, new_actions, num_discrete_actions):
    return ((F.cosine_similarity(old_actions, new_actions)+1)/2).mean().item()
    """Calculate similarity of Continues actions"""
    # Calculate min and max values across both old and new actions for consistency
    # print(old_actions.shape, new_actions.shape)
    min_actions = torch.min(torch.cat([old_actions, new_actions], dim=0), dim=0, keepdim=True)[0]
    max_actions = torch.max(torch.cat([old_actions, new_actions], dim=0), dim=0, keepdim=True)[0]
    
    # Create bins for quantization
    bins = torch.linspace(0, 1, num_discrete_actions + 1, device=old_actions.device)
    
    def convert_to_discrete_one_hot(actions):
        # Normalize actions based on the shared min and max values
        normalized_actions = (actions - min_actions) / (max_actions - min_actions + 1e-8)
        
        # Quantize the normalized actions
        discrete_actions = torch.bucketize(normalized_actions, bins, right=True) - 1
        discrete_actions = torch.clamp(discrete_actions, 0, num_discrete_actions - 1)

        # Convert to one-hot encoding and flatten, cast to float
        one_hot_actions = F.one_hot(discrete_actions, num_classes=num_discrete_actions).float()
        return one_hot_actions.view(actions.size(0), -1)

    # Convert both old and new actions to one-hot encoded representations
    old_one_hot = convert_to_discrete_one_hot(old_actions)
    new_one_hot = convert_to_discrete_one_hot(new_actions)
    
    # Calculate cosine similarity between the two sets of actions
    similarity = F.cosine_similarity(old_one_hot, new_one_hot, dim=1)  # (batch,)
    
    # Adjust the similarity range from [-1, 1] to [0, 1] and return the mean similarity
    adjusted_similarity = (similarity + 1) / 2
    mean_similarity = adjusted_similarity.mean().item()
    
    return mean_similarity

class Cohort:
    def __init__(self):
        self.models: List[nn.Sequential] = []
        self.g: List[float] = []
        self.b: List[float] = []

    def __len__(self):
        return len(self.models)
    
    def append(self, g, b, model):
        self.models.append(model)
        self.g.append(g)
        self.b.append(b)

    def clear(self):
        self.models.clear()
        self.g.clear()
        self.b.clear()

    def similar(self) -> nn.Sequential:
        """min max normalization"""
        g_max = max(self.g)
        g_min = min(self.g)
        b_max = max(self.b)
        b_min = min(self.b)
        similarity = [(((g-g_min)/(g_max-g_min+1e-8)) + ((b-b_min)/(b_max-b_min+1e-8))/2) for g, b in zip(self.g, self.b)]
        idx = similarity.index(max(similarity))
        return self.models[idx]

class ReplayBuffer:
    def __init__(self, buffer_size: int = 0, steps: int = 0, dilation: int = 0, discrete_action: bool = False, 
                 similarity_threshold: float = 1.0, similarity_cohort: int = 1, accumulative_reward=False, 
                 prioritize: bool = True, patience=10):
        
        self.steps: int = steps
        self.dilation: int = dilation
        self.good_buffer: CustomDeque[ContainerBuffer] = CustomDeque(maxlen=buffer_size)
        self.bad_buffer: CustomDeque[ContainerBuffer] = CustomDeque(maxlen=buffer_size)
        self.state: Deque[torch.Tensor] = deque(maxlen=steps+dilation)
        self.action: Deque[torch.Tensor] = deque(maxlen=steps+dilation)
        self.reward: Deque[int] = deque(maxlen=steps+dilation)

        self.discrete_action: bool = discrete_action
        self.similarity_threshold: float = similarity_threshold
        self.similarity_cohort: int = similarity_cohort
        self.offsprings: Cohort = Cohort()

        self.upper_bound = None
        self.lower_bound = None
        self.currant_action = Action.Normal

        self.accumulative_reward = accumulative_reward
        self.prioritize = prioritize
        self.patience = patience

    def is_available(self) -> bool:
        return self.good_buffer.maxlen > 0 and self.steps > 0

    @property
    def size(self) -> int:
        return self.good_buffer.maxlen
    
    def new(self):
        self.state.clear()
        self.action.clear()
        self.reward.clear()
        self.currant_action = Action.Normal


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


    def track(self, state, action, reward):
        while len(self.state) < self.state.maxlen:
            self.step(state, action, reward)
            self.currant_action = Action.Normal
        if self.upper_bound is None:
            self.upper_bound = reward
            self.lower_bound = reward

        self.step(state, action, reward)
        reward = self.reward[-1] if self.accumulative_reward else sum(self.reward)
        if reward >= self.upper_bound:
            self.upper_bound = reward
            if self.currant_action != Action.Good: 
                self.signal(Action.Good, reward)
            self.currant_action = Action.Good
        elif reward <= self.lower_bound:
            self.lower_bound = reward
            if self.currant_action != Action.Bad:
                self.signal(Action.Bad, reward)
            self.currant_action = Action.Bad
        else:
            self.currant_action = Action.Normal

        if self.prioritize:
            self.half_clear()


    def _validate(self, model: nn.Sequential) -> Tuple[float, float]:
        g_similarity, b_similarity = 0, 0
        if len(self.good_buffer) > 0:
            good_new_action: torch.Tensor = model(self.good_buffer.state)
            if self.discrete_action:
                g_similarity = discrete_similarity(self.good_buffer.action, good_new_action.argmax(1), good_new_action.shape[1])# * len(self.good_buffer)
            else:
                g_similarity = continues_similarity(self.good_buffer.action.squeeze(), good_new_action, good_new_action.shape[1])# * len(self.good_buffer)
        if len(self.bad_buffer) > 0:
            bad_new_action: torch.Tensor = model(self.bad_buffer.state)
            if self.discrete_action:
                b_similarity = (1-discrete_similarity(self.bad_buffer.action, bad_new_action.argmax(1), bad_new_action.shape[1]))# * len(self.bad_buffer)
            else:
                b_similarity = (1-continues_similarity(self.bad_buffer.action.squeeze(), bad_new_action, bad_new_action.shape[1]))# * len(self.bad_buffer)
        if len(self) > 0:
            return (g_similarity, b_similarity)
            # return similarity
        else:
            return (1.0, 1.0)
        

    @torch.no_grad()
    def validate(self, model: nn.Sequential) -> nn.Sequential:
        if len(self.offsprings) < self.similarity_cohort:
            g_similarity, b_similarity = self._validate(model)
            if (g_similarity >= self.similarity_threshold) and (b_similarity >= self.similarity_threshold):
                self.offsprings.clear()
                return model
            self.offsprings.append(g_similarity, b_similarity, model)
        else:
            model = self.offsprings.similar()
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
    
