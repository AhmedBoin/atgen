import torch
from torch import nn
import gymnasium as gym
from PIL import Image

env = gym.make("CarRacing-v2")
layer = nn.MaxPool2d(2, 2)

state, info = env.reset()
for i in range(1000):
    # torch_state = torch.from_numpy(state).permute(2, 0, 1)
    # for _ in range(4):
    #     torch_state = layer(torch_state)
    #     print(torch_state.shape)
    # state = torch_state.permute(1, 2, 0).numpy()
    image = Image.fromarray(state).resize((48, 48)).convert('L')
    image.save(f"./image/{i}.jpg")
    state, reward, terminated, truncated, info = env.step(env.action_space.sample())
    
