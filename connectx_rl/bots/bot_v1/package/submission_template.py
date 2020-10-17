def my_agent(observation, configuration):
    from random import choice
    import numpy as np
    this_coice = 0
    _board_width = 7
    _board_height = 6

    master_truth_table = {REPLACEME}

    
    obs = np.reshape(observation.board, (_board_width, _board_height)).T
    
    if str(obs) in master_truth_table:
        this_choice = master_truth_table[str(obs)]
        #print('chosen')
    else:
        this_choice = choice([c for c in range(_board_width) if observation.board[c] == 0])
        #print('random')
    return this_choice