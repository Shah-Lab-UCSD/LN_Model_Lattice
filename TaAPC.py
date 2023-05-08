# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 12:46:56 2021
TaAPC.py

@author: jwjam
"""
import numpy as np
from Agent import Agent


class TaAPC(Agent):
    def __init__(self):
        super().__init__()
        
        self.cognate = True
        self.color = '#FF69B4'
        
        ########## SIZE & MOVEMENT: ##########
        self.size = 2.0
        self.radius = self.size/2 # DC soma radius
        
        self.v = 0
        
        ########## ACTIVATION: ##########
        self.cognate = True
