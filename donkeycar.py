import gymnasium as gym
import numpy as np

import gym_donkeycar

env = gym.make("donkey-generated-roads-v0")

obs, info = env.reset()
try:
    for _ in range(100):
        # drive straight with small speed
        action = np.array([0.0, 0.5])  
        # execute the action
        obs, reward, terminated, truncated, info = env.step(action)
except KeyboardInterrupt:
    # You can kill the program using ctrl+c
    pass

    # Exit the scene
env.close()