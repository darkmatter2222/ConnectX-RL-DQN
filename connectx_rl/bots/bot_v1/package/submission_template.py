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
    else:
        this_choice = choice([c for c in range(_board_width) if observation.board[c] == 0])
    return this_choice


from kaggle_environments import evaluate, make, utils

env = make("connectx", debug=True)

win = 0
loss = 0

for i in range(100):
    env.reset()
    # Play as the first agent against default "random" agent.
    env.run([my_agent, "random"])
    if env.state[0].reward == 1:
        win += 1
    else:
        loss += 1
    # env.render(mode="ipython", width=500, height=450)

print(f'Win:{win} Loss:{loss}')