# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 16:55:22 2022

@author: davem
"""

import numpy as np
from Agent import Agent
import random as rand

class Knee(Agent):
    def __init__(self, antigen_load):
        super().__init__()
        
        self.antigen = [antigen_load]
        self.Tcell = [0]
        self.Tadd = [0]
        self.intime = 0
        self.drain_rate = 3.2*10**-5
        self.antigen_constant = 0.01
        
    def inflam(self, tau):
        self.intime += 1
        self.antigen = self.antigen + [self.antigen[self.intime - 1] - self.antigen[self.intime - 1]*self.drain_rate*tau + (self.Tcell[self.intime-1])/20]
        
    def update_cells(self, Tcells):
        if self.intime < 60*60*8*3*2/12:
            self.Tcell.append(self.Tcell[-1] + Tcells)
            self.Tadd.append(Tcells)
        
        else:
            self.Tcell.append(self.Tcell[-1] + Tcells - self.Tadd[-7200*2])
            self.Tadd.append(Tcells)
            
    def DC_drain(self,tau):
        if rand.random() <= 1 - np.exp(-tau/60/60*(.3-.3*np.exp(-self.antigen[self.intime]*self.antigen_constant))):
            return True
        else:
            return False
        