from __future__ import annotations
import os
import cv2
import json
import torch
import numpy as np
from typing import Union
from enum import Enum, auto

from scipy import interpolate
from scipy.spatial.transform import Rotation

def gen_cubic_spline_interp_func(c2ws: np.ndarray, smoothing_term=10.0, *args, **kwargs):
    # Split interpolation
    N = len(c2ws)
    assert N > 3, 'Cubic Spline interpolation requires at least four inputs'
    if smoothing_term == 0:
        low = -2  # when we view index as from 0 to n, should remove first two segments
        high = N - 1 + 4 - 2  # should remove last one segment, please just work...
        c2ws = np.concatenate([c2ws[-2:], c2ws, c2ws[:2]])


def gen_linear_interp_func(lins: np.ndarray, smoothing_term=10.0):  # smoothing_term <= will loop the interpolation
    if smoothing_term == 0:
        n = len(lins)
        low = -2  # when we view index as from 0 to n, should remove first two segments
        high = n - 1 + 4 - 2  # should remove last one segment, please just work...
        lins = np.concatenate([lins[-2:], lins, lins[:2]])

    lf = interpolate.interp1d(np.linspace(0, 1, len(lins), dtype=np.float32), lins, axis=-2)  # repeat

    if smoothing_term == 0:
        def pf(us): return lf((us * n - low) / (high - low))  # periodic function will call the linear function
        f = pf  # periodic function
    else:
        f = lf  # linear function
    return f