import numpy as np
import torch

def get_module_by_str(module, key):
    if '.' in key:
        keys = key.split('.')
        return get_module_by_str(getattr(module, keys[0]), '.'.join(keys[1:]))
    else:
        return getattr(module, key, None)