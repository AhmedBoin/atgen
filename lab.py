import gymnasium as gym

# Create a custom Lunar Lander environment
class CustomLunarLander(gym.envs.box2d.LunarLander):
    def __init__(self):
        super().__init__()
        self.gravity = 15.0  # Increase gravity
        self.max_fuel = 50   # Reduce fuel

# Initialize environment
env = CustomLunarLander()

# Example of training loop
for episode in range(100):
    observation = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()  # Replace with your agent's action
        observation, reward, done, info = env.step(action)
        env.render()  # Render the environment