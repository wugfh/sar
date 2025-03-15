import scipy.io as sci
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import sys
from scipy import signal
sys.path.append(r"./")
from sinc_interpolation import SincInterpolation

class AutoFocus:
    