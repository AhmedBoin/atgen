from collections import deque
from typing import List
import torch
from torch import nn
import torch.nn.functional as F
import math


class Action:
    Good = "good"
    Bad = "bad"
    Normal = "normal"


class ContainerBuffer:
    def __init__(self, state, action, reward=None, action_type=Action.Normal):
        self.state: torch.Tensor = torch.concat(tuple(state), dim=0)
        self.action: torch.Tensor = torch.stack(tuple(action), dim=0)
        self.reward: torch.Tensor = reward if reward is None else torch.tensor(reward)
        self.action_type = action_type

    def __str__(self) -> str:
        return f"Container(state: {self.state}, action: {self.action}, reward: {self.reward}, type: {self.action_type})"


def discrete_distance(old_actions: torch.Tensor, new_actions: torch.Tensor, actions: int, method='zscore') -> float:
    """Calculate Hamming Distance of discrete actions"""
    # 1. Convert to one-hot vectors
    old_actions = F.one_hot(old_actions, num_classes=actions).float()
    new_actions = F.one_hot(new_actions, num_classes=actions).float()
    # 2. Apply Exponential Moving Average (EMA) smoothing
    smoothing_factor = 1/actions
    old_smoothed = torch.zeros_like(old_actions)
    old_smoothed[0] = old_actions[0]
    for i in range(1, len(old_actions)):
        old_smoothed[i] = smoothing_factor * old_actions[i] + (1 - smoothing_factor) * old_smoothed[i - 1]
    new_smoothed = torch.zeros_like(new_actions)
    new_smoothed[0] = new_actions[0]
    for i in range(1, len(new_actions)):
        new_smoothed[i] = smoothing_factor * new_actions[i] + (1 - smoothing_factor) * new_smoothed[i - 1]
    # 3. Calculate Euclidean distance between two sequences
    return continues_distance(old_smoothed, new_smoothed, method)


def continues_distance(batch1: torch.Tensor, batch2: torch.Tensor, method='zscore', scale_range=(-1, 1)) -> float:
    """Calculate Euclidean Distance of Continues actions"""
    combined_batches = torch.cat([batch1, batch2], dim=0)
    
    if method == 'minmax':
        min_scale, max_scale = min(scale_range), max(scale_range)
        min_val = combined_batches.min(dim=0, keepdim=True)[0]
        max_val = combined_batches.max(dim=0, keepdim=True)[0]
        normalized_batch1 = min_scale + (batch1 - min_val) * (max_scale - min_scale) / (max_val - min_val)
        normalized_batch2 = min_scale + (batch2 - min_val) * (max_scale - min_scale) / (max_val - min_val)
    elif method == 'zscore':
        mean = combined_batches.mean(dim=0, keepdim=True)
        std = combined_batches.std(dim=0, keepdim=True)
        normalized_batch1 = (batch1 - mean) / std
        normalized_batch2 = (batch2 - mean) / std
    else:
        raise ValueError(f"Unknown normalizing method: {method}")
    
    return (F.cosine_similarity(normalized_batch1, normalized_batch2).mean().item() + 1) / 2


class ReplayBuffer:
    def __init__(self, buffer_size: int=0, steps: int = 0, discrete_action: bool = False, 
                 threshold: float = 0.3, method='zscore', scale_range=(-1, 1), reward_range=(0, 0),
                 average_distance: bool = True, prioritize: bool = False):
        
        self.buffer_size = buffer_size
        self.steps = steps
        self.buffer: List[ContainerBuffer] = deque(maxlen=buffer_size)
        self.state: List[torch.Tensor] = deque(maxlen=steps)
        self.action: List[torch.Tensor] = deque(maxlen=steps)
        self.reward: List[int] = deque(maxlen=steps)

        self.discrete_action = discrete_action
        self.threshold = threshold
        self.method = method
        self.scale_range = scale_range

        self.upper_bound = max(reward_range)
        self.lower_bound = min(reward_range)
        self.currant_action = Action.Normal

        self.average_distance = average_distance
        self.prioritize = prioritize
        self._prepared = False

    def step(self, state, action, reward=None):
        self.state.append(state)
        self.action.append(action)
        if self.prioritize:
            self.reward.append(reward)

    def signal(self, action_type: str):
        if self.prioritize:
            if len(self.buffer) == self.buffer_size:
                reward, idx = sum(self.reward), None
                for i, container in enumerate(self.buffer):
                    container: ContainerBuffer

                    if action_type == Action.Good:
                        if sum(container.reward) < reward:
                            reward, idx = sum(container.reward), i
                    elif action_type == Action.Bad:
                        if sum(container.reward) > reward:
                            reward, idx = sum(container.reward), i

                if idx is not None:
                    self.buffer[idx] = ContainerBuffer(self.state, self.action, self.reward, action_type)
            else:
                self.buffer.append(ContainerBuffer(self.state, self.action, self.reward, action_type))
        else:
            self.buffer.append(ContainerBuffer(self.state, self.action, self.reward, action_type))

    def track(self, state, action, reward):
        while len(self.state) < self.steps:
            self.step(state, action, reward)
            self.currant_action = Action.Normal

        self.step(state, action, reward)
        if reward > self.upper_bound:
            if self.currant_action != Action.Good: 
                self.signal(Action.Good)
            self.currant_action = Action.Good
        elif reward < self.lower_bound:
            if self.currant_action != Action.Bad:
                self.signal(Action.Bad)
            self.currant_action = Action.Bad
        else:
            self.currant_action = Action.Normal
        

    @torch.no_grad()
    def validate(self, model: nn.Sequential) -> bool:
        device = model.parameters().__next__().device
        if self.average_distance:
            good_state, good_action, bad_state, bad_action = [], [], [], []
            for container in self.buffer:
                if container.action_type == Action.Good:
                    good_state.append(container.state)
                    good_action.append(container.action)
                if container.action_type == Action.Bad:
                    bad_state.append(container.state)
                    bad_action.append(container.action)
            g_distance, b_distance = 0, 0
            if good_state:
                good_states = torch.concat(good_state).to(device)
                good_actions = torch.concat(good_action).to(device)
                good_new_action: torch.Tensor = model(good_states)
                if self.discrete_action:
                    g_distance = discrete_distance(good_actions, good_new_action.argmax(1), good_new_action.shape[1], self.method) * len(good_state)
                else:
                    g_distance = continues_distance(good_actions, good_new_action, self.method, self.scale_range) * len(good_state)
            if bad_state:
                bad_states = torch.concat(bad_state).to(device)
                bad_actions = torch.concat(bad_action).to(device)
                bad_new_action: torch.Tensor = model(bad_states)
                if self.discrete_action:
                    b_distance = (1-discrete_distance(bad_actions, bad_new_action.argmax(1), bad_new_action.shape[1], self.method)) * len(bad_state)
                else:
                    b_distance = (1-continues_distance(bad_actions, bad_new_action, self.method, self.scale_range)) * len(bad_state)
            if len(self.buffer) > 0:
                distance = (g_distance + b_distance) / len(self.buffer)
                # distance = 1 - (abs(g_distance - b_distance) / (g_distance + b_distance + 1e-8))
                return distance < self.threshold 
            else:
                return True
        else:
            for container in self.buffer:
                new_action: torch.Tensor = model(container.state)
                if self.discrete_action:
                    distance = discrete_distance(container.action, new_action.argmax(1), new_action.shape[1], self.method)
                else:
                    distance = continues_distance(container.action, new_action, self.method, self.scale_range)
                if (container.action_type == Action.Good) and (distance > self.threshold):
                    return False 
                elif (container.action_type == Action.Bad) and (distance < self.threshold):
                    return False 
            return True 
        
    def clear(self):
        self.buffer.clear()
        self.state.clear()
        self.action.clear()
        self.reward.clear()
        self.currant_action = Action.Normal

    def __len__(self) -> int:
        return len(self.buffer)
    

if __name__ == "__main__":
    
    def test_replay_buffer_initialization():
        buffer = ReplayBuffer(buffer_size=10, steps=5)
        assert len(buffer) == 0, "Buffer should be empty on initialization"
        assert buffer.steps == 5, "Steps should be set correctly"
        assert buffer.buffer_size == 10, "Buffer size should be set correctly"
        print("test_replay_buffer_initialization passed")

    def test_track_rewards():
        buffer = ReplayBuffer(buffer_size=10, steps=3, reward_range=(5, -5))
        state = [torch.rand(1, 4) for _ in range(3)]
        action = [torch.rand(1, 2) for _ in range(3)]
        
        buffer.track(state[0], action[0], reward=1)  # Normal state
        buffer.track(state[1], action[1], reward=6)  # Good action
        buffer.track(state[2], action[2], reward=-10)  # Bad action

        assert len(buffer) == 2, "Buffer should contain 2 entries"
        assert buffer.buffer[0].action_type == Action.Good, "First entry should be Good action"
        assert buffer.buffer[1].action_type == Action.Bad, "Second entry should be Bad action"
        print("test_track_rewards passed")

    def test_signal_method():
        buffer = ReplayBuffer(buffer_size=5, prioritize=True)
        state = [torch.rand(1, 4) for _ in range(3)]
        action = [torch.rand(1, 2) for _ in range(3)]
        
        buffer.step(state[0], action[0], reward=10)
        buffer.step(state[1], action[1], reward=-5)
        buffer.signal(Action.Good)
        
        assert len(buffer) == 1, "Buffer should have 1 entry after signaling"
        assert buffer.buffer[0].action_type == Action.Good, "Entry should be of Good action type"
        print("test_signal_method passed")

    def test_validate():
        buffer = ReplayBuffer(buffer_size=5, steps=2, discrete_action=False)
        state = [torch.rand(1, 4) for _ in range(2)]
        action = [torch.rand(1, 2) for _ in range(2)]
        
        buffer.track(state[0], action[0], reward=6)  # Good action
        buffer.track(state[1], action[1], reward=-3)  # Bad action

        model = nn.Sequential(nn.Linear(4, 2))  # Dummy model
        is_valid = buffer.validate(model)
        
        assert not is_valid, "Validation should fail as model isn't trained"
        print("test_validate passed")

    def test_clear_buffer():
        buffer = ReplayBuffer(buffer_size=5, steps=3)
        state = [torch.rand(1, 4) for _ in range(3)]
        action = [torch.rand(1, 2) for _ in range(3)]
        
        buffer.track(state[0], action[0], reward=1)
        buffer.track(state[1], action[1], reward=6)
        buffer.track(state[2], action[2], reward=-10)
        
        buffer.clear()
        assert len(buffer) == 0, "Buffer should be empty after clearing"
        print("test_clear_buffer passed")

    def test_discrete_distance():
        old_actions = torch.tensor([1, 0, 1, 0, 1])
        new_actions = torch.tensor([1, 1, 1, 0, 1])
        dist = discrete_distance(old_actions, new_actions)
        assert math.isclose(dist, 0.2, rel_tol=1e-5), f"Hamming distance should be 0.2, got {dist}"
        print("test_discrete_distance passed")

    def test_continues_distance_zscore():
        batch1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        batch2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        dist = continues_distance(batch1, batch2, method='zscore')
        assert math.isclose(round(dist, 5), 0.0, rel_tol=1e-5), f"Euclidean distance should be 0.0 for identical batches, got {dist}"
        print("test_continues_distance_zscore passed")

    def test_continues_distance_minmax():
        batch1 = torch.tensor([[1., 2.], [3., 4.]])
        batch2 = torch.tensor([[1., 2.], [3., 4.]])
        dist = continues_distance(batch1, batch2, method='minmax', scale_range=(0, 1))
        assert math.isclose(round(dist, 5), 0.0, rel_tol=1e-5), f"Euclidean distance should be 0.0 for identical batches, got {dist}"
        print("test_continues_distance_minmax passed")

    # Run all the tests
    test_replay_buffer_initialization()
    test_track_rewards()
    test_signal_method()
    test_validate()
    test_clear_buffer()
    test_discrete_distance()
    test_continues_distance_zscore()
    test_continues_distance_minmax()
    
    print("All tests passed!")