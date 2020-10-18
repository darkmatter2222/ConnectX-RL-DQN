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