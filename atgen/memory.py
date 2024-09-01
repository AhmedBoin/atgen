import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, state_size: int, buffer_size: int, batch_size: int = 64, device="cpu"):
        if state_size is None:
            state_size = 1
        self.states = np.zeros([buffer_size, state_size], dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.next_states = np.zeros([buffer_size, state_size], dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.batch_size = batch_size
        self.max_size = buffer_size
        self.current_size = 0
        self.ptr = 0
        self.device = device

    def clear(self):
        self.states = np.zeros_like(self.states)
        self.actions = np.zeros_like(self.actions)
        self.rewards = np.zeros_like(self.rewards)
        self.next_states = np.zeros_like(self.next_states)
        self.dones = np.zeros_like(self.dones)
        self.current_size = 0
        self.ptr = 0

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.current_size = min(self.current_size + 1, self.max_size)

    def sample(self):
        idxs = np.random.choice(self.current_size, size=self.batch_size, replace=False)
        
        states = torch.tensor(self.states[idxs]).float().to(self.device)
        actions = torch.tensor(self.actions[idxs]).long().reshape(-1, 1).to(self.device)
        rewards = torch.tensor(self.rewards[idxs]).float().reshape(-1, 1).to(self.device)
        next_states = torch.tensor(self.next_states[idxs]).float().to(self.device)
        dones = torch.tensor(self.dones[idxs]).float().reshape(-1, 1).to(self.device)
        
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return self.current_size