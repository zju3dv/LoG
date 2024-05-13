import os
import torch

def update_global_variable(global_var, cfg):
    for key, val in cfg.items():
        if isinstance(val, dict):
            cfg[key] = update_global_variable(global_var, val)
        elif isinstance(val, str) and val.startswith('$'):
            print('[Config] replace key', val)
            cfg[key] = global_var[val[1:]]
    return cfg

def load_statedict(ckptname, map_location='cpu'):
    statedict = torch.load(ckptname, map_location=map_location)
    if 'state_dict' in statedict.keys():
        statedict = statedict['state_dict']
    return statedict

import fnmatch
def load_gitignore_rules(src_dir):
    ignore_rules = []
    try:
        with open(os.path.join(src_dir, '.gitignore'), 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    ignore_rules.append(line)
    except FileNotFoundError:
        pass
    return ignore_rules

def should_ignore(path, ignore_rules):
    for rule in ignore_rules:
        if fnmatch.fnmatch(path, rule):
            return True
    return False

def copy_files(src_dir, dst_dir):
    """
    copy files according to the rule of .gitignore
    """
    import shutil
    filenames = []
    ignore_rules = load_gitignore_rules(src_dir)
    for root, dirs, files in os.walk(src_dir, topdown=True):
        dirs[:] = [d for d in dirs if d not in ['.git', 'debug', 'data', 'cache', 'output', 'extension', 'submodules']]
        
        for name in files:
            file_path = os.path.join(root, name)
            rel_path = os.path.relpath(file_path, src_dir)
            if not should_ignore(rel_path, ignore_rules):
                dst_path = os.path.join(dst_dir, rel_path)
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copyfile(file_path, dst_path)
                # print(rel_path)
                filenames.append(file_path)
    return filenames

def copy_git_tracked_files(code_dir, output_base_dir):
    from datetime import datetime
    # make new dir
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(output_base_dir, f"code_backup_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    filenames = copy_files('./', output_dir)
    print(f">>> Code {len(filenames)} files has been copied to {output_dir}")
    return output_dir