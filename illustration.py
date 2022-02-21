import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.io import loadmat
from os.path import join as pjoin

class Illustrations:
    def __init__(self, data_file):
        data_dict = loadmat(pjoin('alltaskinfo', data_file))
