# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 9:30:24 2021
agent.py

@author: James Wang
"""
import numpy as np


class Agent():
    def __init__(self):
        self.error = 0.00001
        self.cognate = False
        self.pos = np.array([0,0,0])
        self.color = '#FFFFFF' 