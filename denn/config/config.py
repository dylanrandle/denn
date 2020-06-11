import os
import yaml

this_dir = os.path.dirname(os.path.abspath(__file__))

def get_config(problem_key):
    """ valid pkeys = EXP, SHO, NLO, POS """
    problem_key = problem_key.strip().lower()
    fname = os.path.join(this_dir, f'{problem_key}.yaml')
    with open(fname, 'r') as f:
        params = yaml.full_load(f)
    return params

def write_config(config_dict, save_path):
    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f)

if __name__ == '__main__':
    print(get_config('exp'))
