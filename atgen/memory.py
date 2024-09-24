from collections import deque
from typing import List
import torch
from torch import nn
import torch.nn.functional as F


class Action:
    Good = "good"
    Bad = "bad"
    Normal = "normal"


class ContainerBuffer:
    def __init__(self, state, action, reward=None, action_type=Action.Normal):
        self.state: torch.Tensor = torch.concat(state, dim=0)
        self.action: torch.Tensor = torch.concat(action, dim=0)
        self.reward: torch.Tensor = reward if reward is None else torch.tensor(reward)
        self.action_type = action_type


def hamming_distance(old_actions: torch.Tensor, new_actions: torch.Tensor) -> float:
    """Calculate Hamming Distance of discrete actions"""
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


class ReplayBuffer:
    def __init__(self, buffer_size: int, steps: int = 5, discrete_action: bool = False, 
                 threshold: float = 0.3, method='zscore', scale_range=(-1, 1), reward_range=(0, 0),
                 average_distance: bool = False, prioritize: bool = False):
        
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

        self.upper_bound = reward_range[0]
        self.lower_bound = reward_range[1]
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
                    if action_type == Action.Bad:
                        if sum(container.reward) > reward:
                            reward, idx = sum(container.reward), i

                if idx is not None:
                    self.buffer.pop(idx)
                    self.buffer.append(ContainerBuffer(self.state, self.action, self.reward, action_type))
            else:
                self.buffer.append(ContainerBuffer(self.state, self.action, self.reward, action_type))
        else:
            self.buffer.append(ContainerBuffer(self.state, self.action,  self.reward, action_type))

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

    def _pre_validate(self):
        self.good_state, self.good_action, self.bad_state, self.bad_action = [], [], [], []
        for container in self.buffer:
            if container.action_type == Action.Good:
                self.good_state.append(container.state)
                self.good_action.append(container.action)
            if container.action_type == Action.Good:
                self.bad_state.append(container.state)
                self.bad_action.append(container.action)
        self._prepared = True
        

    @torch.no_grad()
    def validate(self, model: nn.Sequential) -> bool:
        device = model.parameters().__next__().device
        if self.average_distance:
            if not self._prepared:
                self._pre_validate()
            good_states = torch.concat(self.good_state).to(device)
            good_actions = torch.concat(self.good_action).to(device)
            good_new_action: torch.Tensor = model(good_states)
            bad_states = torch.concat(self.bad_state).to(device)
            bad_actions = torch.concat(self.bad_action).to(device)
            bad_new_action: torch.Tensor = model(bad_states)
            if self.discrete_action:
                distance = (hamming_distance(good_actions, good_new_action.argmax(1)) * len(self.good_state) +
                            hamming_distance(bad_actions, bad_new_action.argmax(1)) * len(self.bad_state)) / len(self)
            else:
                distance = (euclidean_distances(good_actions, new_action, self.method, self.scale_range) * len(self.good_state) + 
                            euclidean_distances(bad_actions, new_action, self.method, self.scale_range) * len(self.bad_state)) / len(self)
            return distance > self.threshold
        else:
            for container in self.buffer:
                new_action: torch.Tensor = model(container.state)
                if self.discrete_action:
                    distance = hamming_distance(container.action, new_action.argmax(1))
                else:
                    distance = euclidean_distances(container.action, new_action, self.method, self.scale_range)
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
        self.good_state, self.good_action, self.bad_state, self.bad_action = [], [], [], []
        self.currant_action = Action.Normal

    def __len__(self) -> int:
        return len(self.buffer)
    

if __name__ == "__main__":
    # Test replay buffer initialization
    def test_replay_buffer_initialization():
        buffer = ReplayBuffer(buffer_size=10, steps=5)
        assert len(buffer) == 0, "Buffer should be empty on initialization"
        assert buffer.steps == 5, "Steps should be set correctly"
        assert buffer.buffer_size == 10, "Buffer size should be set correctly"
        print("test_replay_buffer_initialization passed")

    # Test adding steps and tracking rewards
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

    # Test signaling method
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

    # Test validation method with dummy model (identity model)
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

    # Test clear buffer
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

    # Test hamming distance
    def test_hamming_distance():
        old_actions = torch.tensor([1, 0, 1, 0, 1])
        new_actions = torch.tensor([1, 1, 1, 0, 1])
        dist = hamming_distance(old_actions, new_actions)
        assert dist == 0.2, f"Hamming distance should be 0.2, got {dist}"
        print("test_hamming_distance passed")

    # Test Euclidean distances with zscore normalization
    def test_euclidean_distances_zscore():
        batch1 = torch.tensor([[1., 2.], [3., 4.]])
        batch2 = torch.tensor([[1., 2.], [3., 4.]])
        dist = euclidean_distances(batch1, batch2, method='zscore')
        assert dist == 0.0, f"Euclidean distance should be 0.0 for identical batches, got {dist}"
        print("test_euclidean_distances_zscore passed")

    # Test Euclidean distances with min-max normalization
    def test_euclidean_distances_minmax():
        batch1 = torch.tensor([[1., 2.], [3., 4.]])
        batch2 = torch.tensor([[1., 2.], [3., 4.]])
        dist = euclidean_distances(batch1, batch2, method='minmax', scale_range=(0, 1))
        assert dist == 0.0, f"Euclidean distance should be 0.0 for identical batches, got {dist}"
        print("test_euclidean_distances_minmax passed")

    # Run all the tests
    test_replay_buffer_initialization()
    test_track_rewards()
    test_signal_method()
    test_validate()
    test_clear_buffer()
    test_hamming_distance()
    test_euclidean_distances_zscore()
    test_euclidean_distances_minmax()
    
    print("All tests passed!")