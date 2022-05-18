# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 12:46:11 2021
DC.py

@author: jwjam
"""
import numpy as np
import random as rand
from Agent import Agent


class DC(Agent):
    def __init__(self):
        super().__init__()
        
        
        self.color = '#A31800'
        self.color_act = '#FFB7AB'
        
        self.color_cog = '#006b0a'
        self.color_act_cog = '#abffb3'
        
        ########## SIZE & MOVEMENT: ##########
        self.size = 10
        self.radius_soma = self.size/2 # DC soma radius
        self.radius = 21 # active radius
        
        self.v = 0
        self.cognate = True
        
        ########## ACTIVATION: ##########
        self.contact_num = 0
        self.is_alive = True
        
        self.lifespan = np.random.normal(44, 6)*60*60

    

        
                    
