#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 08:52:22 2021

@author: c1ph3r
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_log_error, mean_absolute_error

# from keras.models import Model
# from keras.layers import LSTM, TimeDistributed, Input, Dense
# from keras.layers.core import Lambda
# from keras import backend as K
# from keras.utils import plot_model

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # or any {'0', '1', '2'}
import warnings
warnings.filterwarnings('ignore')

'''
Level | Level for Humans | Level Description
 -------|------------------|------------------------------------
  0     | DEBUG            | [Default] Print all messages
  1     | INFO             | Filter out INFO messages
  2     | WARNING          | Filter out INFO & WARNING messages
  3     | ERROR            | Filter out all messages
'''

def set_size(width, fraction=1):
    '''
    set aesthetic figure dimensions to avoid scaling in latex.

    Parameters
    ----------
    width: float
            width in points
    fraction: float
            fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of the figure in inches
    '''
    # Width of the figure
    fig_width_pt = width * fraction

    # Convert from point to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 -1) /2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    figure_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, figure_height_in)

    return fig_dim
