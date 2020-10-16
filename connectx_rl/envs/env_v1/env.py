import abc
import tensorflow as tf
import numpy as np
import json
import random

from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
from connectx_rl.bots.bot_v1.package import connectxv1

class env(py_environment.PyEnvironment):
    def __init__(self, env_name, render_me=True, enemy='random'):

        self.env_name = env_name
        self.master_truth_table = {}
        self.last_state = None

        self.step_count = 0
        self.enemy = enemy
        self.state_action_history = {}

        self.state_pos = 0

        self._board_width = 7
        self._board_height = 6
        self._network_frame_depth = 1
        self._channels = 1

        # initialize game
        self.new_environment()

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=self._board_width, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self._network_frame_depth, self._channels,  self._board_height, self._board_width), dtype=np.float,
            minimum=0.0, maximum=1.0, name='observation')

        self.state = np.zeros([self._channels,  self._board_height, self._board_width])
        self.state_history = [self.state] * self._network_frame_depth

        self.episode_ended = True


    def action_spec(self):
        return_object = self._action_spec
        return return_object

    def observation_spec(self):
        return_object = self._observation_spec
        return return_object

    def _reset(self):
        self.state = np.zeros([self._channels,  self._board_height, self._board_width])
        self.state_history = [self.state] * self._network_frame_depth
        self.state_action_history = {}

        self.new_environment()

        obs = self.trainer.reset()
        state = self.obs_to_state(obs)
        self.last_state = state

        self.step_count = 0

        return_object = ts.restart(np.array(self.state_history, dtype=np.float))
        return return_object

    def _step(self, action):
        if self.episode_ended:
            self.reset()

        self.step_count += 1
        int_action = int(action)

        if self.env_name == 'Testing':
            self.state_action_history[str(self.last_state)]  = int_action
        obs, reward, self.episode_ended, info = self.trainer.step(int_action)
        if reward is None:
            reward = 0

        state = self.obs_to_state(obs)
        self.last_state = state

        self.state_history.append([state])
        del self.state_history[:1]

        # ===return to engine===
        if self.episode_ended:
            if reward == 1:
                reward = 24 - self.step_count
                if self.env_name == 'Testing':
                    for state in self.state_action_history:
                        self.master_truth_table[state] = self.state_action_history[state]
            return_object = ts.termination(np.array(self.state_history, dtype=np.float), reward)
            return return_object
        else:
            return_object = ts.transition(np.array(self.state_history, dtype=np.float), reward=reward, discount=1.0)
            return return_object

    def obs_to_state(self, obs):
        return np.reshape(obs['board'], (self._board_width, self._board_height)).T

    def new_environment(self):
        self.environment = make("connectx")
        if self.enemy != 'random':
            if self.enemy == 'connectxv1':
                self.environment.agents[self.enemy] = connectxv1.my_agent

        if random.choice(range(2)) == 0:
            self.trainer = self.environment.train([None, self.enemy])
            self.state_pos = 0
        else:
            self.trainer = self.environment.train([self.enemy, None])
            self.state_pos = 1