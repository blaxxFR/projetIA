# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 23:29:46 2022

@author: Nathan
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from generate_data_learn_V1 import myDataset,createdata
from itertools import chain
import csv

freq = pd.read_csv(open("test.csv", "r"),
                    delimiter=",")
inputs = pd.read_csv(open("dict.csv", "r"),
                    delimiter=",")

freq = freq.transpose()

freq.plot()