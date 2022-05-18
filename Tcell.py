# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 12:48:36 2021
Tcell.py

@author: jwjam
"""
import numpy as np
from Agent import Agent


class Tcell(Agent):
    def __init__(self):
        super().__init__()
        
        self.color_cog = '#9400FF' # Purple
        self.color_noncog = '#D7BDE2'
        self.color_dead = '#4A235A'

        ########## SIZE AND MOVEMENT: ##########
        self.pos = np.array([0,0,0])
        self.size = 6
        self.radius = self.size/2
        
        self.v0 = 1 # initial velocity in m/s
        self.v = 0 # velocity in m/frame; set this during inititalization
        
        # Direction angles in spherical coordinates
        self.phi = 0
        self.theta = 0
        self.momentum = np.pi/6 # resistance to change in movement direction
        
        self.walk = True # True if undergoing random (Brownian) walk
        
        ########## STATUSES: ##########
        self.lifespan = 60*60*24*10
        self.age = 0
        
        self.alive = True
        self.death_t = 0 # time since cell death
        self.death_tlim = 60*60*8 # time since cell death until cell is cleared from simulation
        
        ########## ACTIVATION AND EXPANSION: ##########
        self.mature = False
        self.cognate = True
        self.PD1 = 0 # PD1 expression arbitrary value
        self.avidity = 0
        self.S = 0
        self.E1 = 100 #Activation threshold to swithc to phase 2
        self.E2 = 200 #Activation threshold to switch to phase 3
        self.E3 = 300 #Activiation threshold to begin proliferating
        self.E4 = 100 #Threshold to continue to proliferate

        # self.contact_chemo = 0  # number of times in contact with chemokine gradient (per occasion)
        self.contact_DC = 0     # number of times in contact with DC surface (per occasion)
        self.contact_Tcell = 0  # number of times in contact with T cell
        self.contact_TaAPC = 0  # number of times in contact with TaAPC
        
        # self.chemo_status = False           # whether T cell is in chemokine gradient or not
        # self.chemo_t = 0                    # time undergoing chemowalk
        # self.tlim_chemo_noncog:3*60         # time limit for noncognate T cell to become desensitized in chemokine gradient
        # self.tlim_chemo_cog = 11*60         # time limit for cognate T cell to become desensitized in chemokine gradient
        
        self.phase = 0
        
        self.contact_t = 0              # time in contact with neighbor
        self.contact_tlim = 0           # maximum time in contact and interacting with neighbor
        self.contact_status = False     # currently interacting with neighbor?
        self.contact_can = True         # can interact with neighbors?
        self.contact_can_cdlim = 60*5   # total cooldown time for contacts to occur again
        self.contact_can_cd = self.contact_can_cdlim # current cooldown time for able to contact again; count down
        
        self.tlim_p1_cog = [7, 2]*60            # P1 cognate T cell contact time limits: 3-11 min
        self.tlim_p1_noncog = [4, 0.25]*60      # P1 noncognate T cell contact time limits: 3-5 min
        self.tlim_p2_cog = [12, 2]*60*60           # P2 cognate T cell contact time limits: 8-16 hr
        self.tlim_p2_noncog = [8.5, 2.25]*60    # P2 noncognate T cell contact time limits: 4-13 min
        self.tlim_p3_cog = [7, 2]*60            # P3 cognate T cell contact time limits: 3-11 min
        self.tlim_p3_noncog = [4, 0.25]*60      # P3 noncognate T cell contact time limits: 3-5 min
        
        self.tlim_supp_cog = [7, 2]*60          # cognate T cell-TaAPC contact time
        self.tlim_supp_noncog = [4, 0.25]*60    # noncognate T cell-TaAPC contact time
        
        self.div_ability = False # whether T cell can divide; requires stimulation event during phase 2
        self.div_status = False # whether T cell is dividing or not
        self.div_n = 0          # number of divisions
        self.div_nlim = 8       # T cell can divide up to 8 times
        self.div_t = 0          # time since T cell started dividing
        self.div_tlim = 60*60*np.random.normal(18,1.5) # time it takes for T cell to divide
        self.div_can = True     # T cell can or cannot divide based on minimum time between divisions
        self.div_can_cdlim = 60*60*5   # total cooldown time for divisions to start occuring again
        self.div_can_cd = self.div_can_cdlim # current cooldown time for able to divide again; count down
        
        
        self.stim_p1 = 0 # number of times successfully stimulated in P1
        self.stim_p2 = 0 # " in P2
        self.stim_p3 = 0 # " in P3
        
        self.recorded_data = {'contact_chemo':[],   # number of times in contact with chemokine gradient (per occasion)
                              'contact_DC':[],      # number of times in contact with DC surface (per occasion)
                              'contact_Tcell':[],   # number of times in contact with T cell
                              'contact_TaAPC':[],   # number of times in contact with TaAPC
                              'n_stim_p1':[],       # number of times successfully stimulated in P1
                              'n_stim_p2':[],       # " in P2
                              'n_stim_p3':[],       # " in P3
                              'n_supp':[]}          # number of times successfully suppressed
    
    ## setters
    
    def set_avidity(self, mean = 1, shape = 1.1):
        self.avidity = np.random.lognormal(mean, np.log(shape))
        
    def stimulate_T(self):
        self.S += self.avidity
    
    def move_T(self, all_pos):
        x = int(self.pos[0])
        y = int(self.pos[1])
        z = int(self.pos[2])
        local_fill = np.copy(all_pos[x-2:x+3,y-2:y+3,z-2:z+3])
        num_pos = np.sum(local_fill)
        if num_pos == 0:
            return local_fill
        else:
            index = np.random.randint(num_pos)
            local_inds = np.where(local_fill == 1)
            local_fill[2,2,2] = 1
            x_shift = local_inds[0][index]
            y_shift = local_inds[1][index]
            z_shift = local_inds[2][index]
            
            x += x_shift - 2
            y += y_shift - 2
            z += z_shift - 2
            self.pos = np.array([x,y,z])
            local_fill[x_shift,y_shift,z_shift] = 0
            return local_fill
        
    