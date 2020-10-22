import random
import numpy as np

EMPTY= 0
def is_win(board, column, mark, config, has_played=False):
    columns = config.columns
    rows = config.rows
    inarow = config.inarow - 1

    res = 0
    if has_played:
        this_res = [r for r in range(rows) if board[column + (r * columns)] == mark]
        if len(this_res) > 0:
            res = min(this_res)
        else:
            return False
    else:
        this_res = [r for r in range(rows) if board[column + (r * columns)] == EMPTY]
        if len(this_res) > 0:
            res = max(this_res)
        else:
            return False

    row = (res)

    def count(offset_row, offset_column):
        for i in range(1, inarow + 1):
            r = row + offset_row * i
            c = column + offset_column * i
            if (
                r < 0
                or r >= rows
                or c < 0
                or c >= columns
                or board[c + (r * columns)] != mark
            ):
                return i - 1
        return inarow

    return (
        count(1, 0) >= inarow  # vertical.
        or (count(0, 1) + count(0, -1)) >= inarow  # horizontal.
        or (count(-1, -1) + count(1, 1)) >= inarow  # top left diagonal.
        or (count(-1, 1) + count(1, -1)) >= inarow  # top right diagonal.
    )

def my_agent(observation, configuration):
    thischoice = random.choice([c for c in range(7) if observation.board[c] == 0])

    my_mark = observation.mark
    enemy_mark = 1
    if my_mark == 1:
        enemy_mark = 2

    # is the a win for me?
    for _ in range(7):
        result = is_win(observation.board, _, my_mark, configuration)
        if result:
            return _
    # can I block the enemy?
    for _ in range(7):
        result = is_win(observation.board, _, enemy_mark, configuration)
        if result:
            return _

    return thischoice