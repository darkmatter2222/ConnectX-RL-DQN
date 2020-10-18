import sys
from os.path import dirname, abspath, join

THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, '..', '..', '..'))
THIS_DIR = abspath(join(THIS_DIR))
sys.path.append(THIS_DIR)
sys.path.append(CODE_DIR)
import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from connectx_rl.envs.env_v1.env import env
from tf_agents.networks import categorical_q_network
from tf_agents.agents.categorical_dqn import categorical_dqn_agent
import os
import cv2
import json
from connectx_rl.bots.bot_v1.helpers import helpers
from tf_agents.policies import policy_saver
import numpy as np
import socket
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

_config = helpers.load_configuration()

# set tensorflow compatibility
tf.compat.v1.enable_v2_behavior()

# setting hyperparameters
num_iterations = 150000000 # @param {type:"integer"}

initial_collect_steps = 100  # @param {type:"integer"}
collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_capacity = 100000  # @param {type:"integer"}

reward_history = []
loss_history = []

fc_layer_params = (1000,)

batch_size = 128  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
gamma = 0.99
log_interval = 200  # @param {type:"integer"}

num_atoms = 51  # @param {type:"integer"}
min_q_value = -1  # @param {type:"integer"}
max_q_value = 24  # @param {type:"integer"}
n_step_update = 4  # @param {type:"integer"}

num_eval_episodes = 1000  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}

_num_save_episodes = 10000

reward_history = []
loss_history = []

if not os.path.exists(_config['master_truth_dir']):
    os.makedirs(_config['master_truth_dir'])

if not os.path.exists(_config['master_truth_file']):
    f = open(_config['master_truth_file'], 'w+')  # open file in append mode
    f.write('{}')
    f.close()

# instantiate two environments. I personally don't feel this is necessary,
# however google did it in their tutorial...
_train_py_env = env(env_name='Training', enemy=['random', 'submissionv4', 'submissionv5'])
_eval_py_env = env(env_name='Testing', enemy=['random', 'submissionv4', 'submissionv5'])

train_env = tf_py_environment.TFPyEnvironment(_train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(_eval_py_env)

categorical_q_net = categorical_q_network.CategoricalQNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    num_atoms=num_atoms,
    fc_layer_params=fc_layer_params)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

train_step_counter = tf.compat.v2.Variable(0)

agent = categorical_dqn_agent.CategoricalDqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    categorical_q_network=categorical_q_net,
    optimizer=optimizer,
    min_q_value=min_q_value,
    max_q_value=max_q_value,
    n_step_update=n_step_update,
    td_errors_loss_fn=common.element_wise_squared_loss,
    gamma=gamma,
    train_step_counter=train_step_counter)
agent.initialize()

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def render_history():
    figure, axes = plt.subplots(2)
    canvas = FigureCanvas(figure)

    axes[0].plot(reward_history, 'red')
    axes[0].plot(smooth(reward_history, 4), 'orange')
    axes[0].plot(smooth(reward_history, 8), 'yellow')
    axes[0].plot(smooth(reward_history, 16), 'green')
    axes[0].plot(smooth(reward_history, 32), 'blue')
    axes[0].plot(smooth(reward_history, 64), 'purple')
    canvas.draw()
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8')

    img = image.reshape(canvas.get_width_height()[::-1] + (3,))

    # img is rgb, convert to opencv's default bgr
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # display image with opencv or any operation you like
    cv2.imshow("plot", img)
    cv2.waitKey(1)
    plt.close('all')
# collect_data(train_env, random_policy, replay_buffer, steps=100)

random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())

helpers.compute_avg_return(eval_env, random_policy, num_eval_episodes)


replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_capacity)

def collect_step(environment, policy):
  time_step = environment.current_time_step()
  action_step = policy.action(time_step)
  next_time_step = environment.step(action_step.action)
  traj = trajectory.from_transition(time_step, action_step, next_time_step)

  # Add trajectory to the replay buffer
  replay_buffer.add_batch(traj)

for _ in range(initial_collect_steps):
  collect_step(train_env, random_policy)

# This loop is so common in RL, that we provide standard implementations of
# these. For more details see the drivers module.

# Dataset generates trajectories with shape [BxTx...] where
# T = n_step_update + 1.
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3, sample_batch_size=batch_size,
    num_steps=n_step_update + 1).prefetch(3)

iterator = iter(dataset)

train_checkpointer = common.Checkpointer(
    ckpt_dir=_config['checkpoint_policy_dir'],
    max_to_keep=1,
    agent=agent,
    policy=agent.policy,
    replay_buffer=replay_buffer,
    global_step=train_step_counter
)

tf_policy_saver = policy_saver.PolicySaver(agent.policy)

restore_network = True

if restore_network:
    train_checkpointer.initialize_or_restore()
    f = open(_config['master_truth_file'], "r")
    eval_env.pyenv._envs[0].master_truth_table = json.loads(f.read())
    f.close()

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = helpers.compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]

for _ in range(num_iterations):

  # Collect a few steps using collect_policy and save to the replay buffer.
  for _ in range(collect_steps_per_iteration):
    collect_step(train_env, agent.collect_policy)

  # Sample a batch of data from the buffer and update the agent's network.
  experience, unused_info = next(iterator)
  train_loss = agent.train(experience)

  step = agent.train_step_counter.numpy()

  if step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, train_loss.loss))

  if step % eval_interval == 0:
    avg_return, results, enemy_history = helpers.compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    reward_history.append(avg_return)
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
    print('step = {0}: Average Return = {1:.2f}'.format(step, avg_return))
    render_history()
  if step % _num_save_episodes == 0:
    tf_policy_saver.save(_config['save_policy_dir'])
    train_checkpointer.save(train_step_counter)
    #print(f'Saving truth table of length {len(eval_env.pyenv._envs[0].master_truth_table.keys())}')
    #f = open(_master_truth_file, "w")
    #f.write(json.dumps(eval_env.pyenv._envs[0].master_truth_table))
    #f.close()