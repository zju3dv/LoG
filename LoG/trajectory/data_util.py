import os
import re
import cv2
import h5py
import torch
import struct
import asyncio
import subprocess
import numpy as np

from PIL import Image
from io import BytesIO
from typing import overload
from functools import lru_cache

# from imgaug import augmenters as iaa
from typing import Tuple, Union, List, Dict

from torch.nn import functional as F
from torch.utils.data._utils.pin_memory import pin_memory
from torch.utils.data._utils.collate import default_collate, default_convert

from LoG.trajectory.dotdict import dotdict

def to_numpy(batch, non_blocking=False, ignore_list: bool = False) -> Union[List, Dict, np.ndarray]:  # almost always exporting, should block
    if isinstance(batch, (tuple, list)) and not ignore_list:
        batch = [to_numpy(b, non_blocking, ignore_list) for b in batch]
    elif isinstance(batch, dict):
        batch = dotdict({k: to_numpy(v, non_blocking, ignore_list) for k, v in batch.items()})
    elif isinstance(batch, torch.Tensor):
        batch = batch.detach().to('cpu', non_blocking=non_blocking).numpy()
    else:  # numpy and others
        batch = np.asarray(batch)
    return batch

def to_tensor(batch, ignore_list: bool = False) -> Union[torch.Tensor, dotdict[str, torch.Tensor]]:
    if isinstance(batch, (tuple, list)) and not ignore_list:
        batch = [to_tensor(b, ignore_list) for b in batch]
    elif isinstance(batch, dict):
        batch = dotdict({k: to_tensor(v, ignore_list) for k, v in batch.items()})
    elif isinstance(batch, torch.Tensor):
        pass
    else:  # numpy and others
        batch = torch.as_tensor(batch)
    return batch

def to_cuda(batch, device="cuda", ignore_list: bool = False) -> torch.Tensor:
    if isinstance(batch, (tuple, list)):
        batch = [to_cuda(b, device, ignore_list) for b in batch]
    elif isinstance(batch, dict):
        batch = dotdict({k: (to_cuda(v, device, ignore_list) if k != "meta" else v) for k, v in batch.items()})
    elif isinstance(batch, torch.Tensor):
        batch = batch.to(device, non_blocking=True)
    else:  # numpy and others
        batch = torch.as_tensor(batch, device=device)
    return batch

def to_list(batch, non_blocking=False) -> Union[List, Dict, np.ndarray]:  # almost always exporting, should block
    if isinstance(batch, (tuple, list)):
        batch = [to_list(b, non_blocking) for b in batch]
    elif isinstance(batch, dict):
        batch = dotdict({k: to_list(v, non_blocking) for k, v in batch.items()})
    elif isinstance(batch, torch.Tensor):
        batch = batch.detach().to('cpu', non_blocking=non_blocking).numpy().tolist()
    elif isinstance(batch, torch.Tensor):
        batch = batch.tolist()
    else:  # others, keep as is
        pass
    return batch