import os
import json
import socket


# loading configuration...
print('loading configuration...')
_config = {}
with open('config.json') as f:
    _config = json.load(f)

host_name = socket.gethostname()
base_directory_key = 'base_dir'
target = f'{host_name}-base_dir'
if target in _config['files']['policy']:
    base_directory_key = target

_master_truth_dir = os.path.join(_config['files']['policy'][base_directory_key],
                                      _config['files']['policy']['master_truth']['dir'])

_master_truth_file = os.path.join(_config['files']['policy'][base_directory_key],
                                      _config['files']['policy']['master_truth']['dir'],
                                      _config['files']['policy']['master_truth']['name'])

f = open(_master_truth_file, "r")
payload = f.read()
f.close()

f = open('package//submission_template.py', "r")
sub_template = f.read()
f.close()

f = open('package//connectxv1.py', "w")
f.write(sub_template.replace("{REPLACEME}", payload))
f.close()


