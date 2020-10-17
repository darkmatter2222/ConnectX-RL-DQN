from os.path import dirname, abspath, join
import sys
THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, '..', '..', '..'))
THIS_DIR = abspath(join(THIS_DIR))
sys.path.append(THIS_DIR)
sys.path.append(CODE_DIR)
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from connectx_rl.envs.env_v1.env import env
from tf_agents.networks import categorical_q_network
from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tqdm import tqdm
import os
import cv2
import json
from tf_agents.policies import policy_saver
import numpy as np
import socket
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import tensorflow_addons as tfa

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

_checkpoint_policy_dir = os.path.join(_config['files']['policy'][base_directory_key],
                                      _config['files']['policy']['checkpoint_policy']['dir'],
                                      _config['files']['policy']['checkpoint_policy']['name'])

_master_truth_dir = os.path.join(_config['files']['policy'][base_directory_key],
                                      _config['files']['policy']['master_truth']['dir'])

_master_truth_file = os.path.join(_config['files']['policy'][base_directory_key],
                                      _config['files']['policy']['master_truth']['dir'],
                                      _config['files']['policy']['master_truth']['name'])

if not os.path.exists(_master_truth_dir):
    os.makedirs(_master_truth_dir)

# set tensorflow compatibility
tf.compat.v1.enable_v2_behavior()

# setting hyperparameters
num_iterations = 150000000 # @param {type:"integer"}

initial_collect_steps = 1000  # @param {type:"integer"}
collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_capacity = 100000  # @param {type:"integer"}

fc_layer_params = (1000,)

batch_size = 128  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
gamma = 0.99
log_interval = 200  # @param {type:"integer"}

num_atoms = 51  # @param {type:"integer"}
min_q_value = -1  # @param {type:"integer"}
max_q_value = 24  # @param {type:"integer"}
n_step_update = 2  # @param {type:"integer"}

num_eval_episodes = 1000  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}

_num_save_episodes = 1000

_eval_py_env = env(env_name='Testing', enemy=['random', 'submissionv2', 'negamax'])

eval_env = tf_py_environment.TFPyEnvironment(_eval_py_env)

policy = tf.saved_model.load(_save_policy_dir)

if not os.path.exists(_master_truth_file):
    f = open(_master_truth_file, 'w+')  # open file in append mode
    f.write('{}')
    f.close()
else:
    f = open(_master_truth_file, 'r')  # open file in append mode
    eval_env.pyenv._envs[0].master_truth_table = json.loads(f.read())
    f.close()

def compute_avg_return(environment, policy, num_episodes=10):
  total_return = 0.0

  results = {
      'win': {
          'first': 0,
          'second': 0
      },
      'loss': {
          'first': 0,
          'second': 0
      },
      'tie': {
          'first': 0,
          'second': 0
      }
  }
  enemy_history = {}

  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return
    state_pos = environment.pyenv._envs[0].state_pos
    win_flag = environment.pyenv._envs[0].environment.state[state_pos].reward
    chosen_enemy = environment.pyenv._envs[0].chosen_enemy
    if chosen_enemy in enemy_history:
        enemy_history[chosen_enemy] += 1
    else:
        enemy_history[chosen_enemy] = 1

    if state_pos == 0:
        if win_flag == 1:
            results['win']['first'] += 1
        elif win_flag == -1:
            results['loss']['first'] += 1
        else:
            results['tie']['first'] += 1
    elif state_pos == 1:
        if win_flag == 1:
            results['win']['second'] += 1
        elif win_flag == -1:
            results['loss']['second'] += 1
        else:
            results['tie']['second'] += 1

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0], results, enemy_history

for _ in range(num_iterations):
    avg_return, results, enemy_history = compute_avg_return(eval_env, policy, num_eval_episodes)
    print(f'Eval Going First '
          f'Wins:{results["win"]["first"]} '
          f'Losss:{results["loss"]["first"]} '
          f'Ties:{results["tie"]["first"]} '
          f' Going Second '
          f'Wins:{results["win"]["second"]} '
          f'Losss:{results["loss"]["second"]} '
          f'Ties:{results["tie"]["second"]} '
          f' Enemy History '
          f'{enemy_history}')
    print(f'Saving truth table of length {len(eval_env.pyenv._envs[0].master_truth_table.keys())}')
    f = open(_master_truth_file, "w")
    f.write(json.dumps(eval_env.pyenv._envs[0].master_truth_table))
    f.close()
    #print('step = {0}: Average Return = {1:.2f}'.format(step, avg_return))
    #returns.append(avg_return)