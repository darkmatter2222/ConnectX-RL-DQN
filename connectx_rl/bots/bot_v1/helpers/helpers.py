import os, json, socket

def load_configuration():
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

    config = {}

    config['save_policy_dir'] = os.path.join(_config['files']['policy'][base_directory_key],
                                    _config['files']['policy']['save_policy']['dir'],
                                    _config['files']['policy']['save_policy']['name'])

    config['checkpoint_policy_dir'] = os.path.join(_config['files']['policy'][base_directory_key],
                                          _config['files']['policy']['checkpoint_policy']['dir'],
                                          _config['files']['policy']['checkpoint_policy']['name'])

    config['master_truth_dir'] = os.path.join(_config['files']['policy'][base_directory_key],
                                     _config['files']['policy']['master_truth']['dir'])

    config['master_truth_file'] = os.path.join(_config['files']['policy'][base_directory_key],
                                      _config['files']['policy']['master_truth']['dir'],
                                      _config['files']['policy']['master_truth']['name'])

    config['executable_bots_dir'] = os.path.join(_config['files']['policy'][base_directory_key],
                                        _config['files']['policy']['executable_bots']['dir'])
    return config


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