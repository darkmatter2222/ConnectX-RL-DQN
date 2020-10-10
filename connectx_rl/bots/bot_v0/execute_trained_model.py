import os
import json
import socket
from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *
import tempfile
from tf_agents import policies
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# loading configuration...
print('loading configuration...')
_config = {}
with open('config.json') as f:
    _config = json.load(f)

# build policy directories
host_name = socket.gethostname()
base_directory_key = 'base_dir'
target = f'{host_name}-base_dir'
if target in _config['files']['policy']:
    base_directory_key = target


_save_policy_dir = os.path.join(_config['files']['policy'][base_directory_key],
                                _config['files']['policy']['save_policy']['dir'],
                                _config['files']['policy']['save_policy']['name'])
_board_width = 7
_board_height = 6

def obs_to_state(obs):
    return np.reshape(obs.board, (_board_width, _board_height)).T


saved_policy = tf.compat.v2.saved_model.load(_save_policy_dir)
print(list(saved_policy.signatures.keys()))  # ["serving_default"]

environment = make("connectx")
trainer = environment.train([None, "random"])
obs = trainer.reset()
state = obs_to_state(obs)

def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in tqdm(range(num_episodes)):
        time_step = environment.reset()
        episode_return = 0.0
        while not environment.done:
            state = obs_to_state(time_step[0].observation)
            state2 = np.array([np.array([state])])
            action_step = policy.action(state2)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return
    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

avg_return = compute_avg_return(environment, saved_policy, 10)
print(f'Average Return {avg_return}')