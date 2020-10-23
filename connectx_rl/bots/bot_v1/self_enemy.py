from random import choice
import numpy as np

master_truth_table = {}

def my_agent(observation, configuration):
    this_choice = 0
    _board_width = 7
    _board_height = 6

    obs = np.reshape(observation.board, (_board_width, _board_height)).T
    if str(obs) in master_truth_table:
        this_choice = master_truth_table[str(obs)]
    else:
        this_choice = choice([c for c in range(_board_width) if observation.board[c] == 0])
    return this_choice