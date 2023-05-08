# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 12:55:06 2021
LN.py

@author: James Wang
"""
from Tcell import Tcell
from DC import DC
from TaAPC import TaAPC
from Knee import Knee
import random as rand

import numpy as np
import matplotlib as mpl
import time
import matplotlib.pyplot as plt
from matplotlib import cm
import copy
from mpl_toolkits.mplot3d import Axes3D


class LN():
    def __init__(self, agent_ct, cognate_Tcell, end, vis_res, affinity_scale, affinity_shape = 1.1, antigen_conc_0 = 10**3, seed = 0):
        # Figure 2: vary DC cognate concentration/frequency in cognate_DC
        # Figure 3: vary number of starting T cells in agent_ct <Here I think we rather want to vary cognate T cell fraction
        # Figure 4: introduce TaAPCs into simulation in run_LN.py file 
        
        self.seed = seed
        
        self.tau = 12 # seconds per frame
        self.end = end # time of ending simulation; i.e. 60*60*24*5 (5 days)
        
        self.bounds = 0 # radius of simulation
        self.bounds_int = 0 #number of voxels in a direction
        
        #timing aspects
        self.start_time = 0
        self.real_time = 0
        self.chkpt25 = True
        self.chkpt50 = True
        self.chkpt75 = True
        self.chkpt100 = True
        
        self.div_check_temp = True
        
        self.stimcount = 0
        
        self.run_pct = 0
        self.time = 0
        self.error = 0.00001
        
        self.vis_res = vis_res # i.e. 72 or 300 dpi

        self.agent_ct = agent_ct
        # i.e. agent_ct = {'Tcell':100,
        #                   'DC':5,
        #                   'TaAPC':100}
        
        # Cognate T cell frequency
        self.cognate_Tcell = cognate_Tcell
        self.cognate_Tcell_n = int(np.ceil(self.agent_ct['Tcell'] * self.cognate_Tcell))
        
        #cognate T cell parameters
        self.affinity_scale = affinity_scale
        self.affinity_shape = affinity_shape
        self.Tegress = 0.000017
        
        #Parameters that we want to record
        self.deadT = 0
        self.goneT = 0
        self.kneeT = 0
        self.T_to_DC = 0
        self.T_to_TaAPC = 0
        self.liveT = self.agent_ct['Tcell']
        self.cogT = self.cognate_Tcell_n
        self.cogDCs = 0

        # Set T cell density: percent of total lymph node volume occupied by T cells
        self.density_Tcell = 0.1
        self.all_pos = np.array([])
        self.all_soma = np.array([])
        self.one_soma = np.array([])
        self.vessels = np.array([])
        self.all_taapc = np.array([])
        
        self.ID_max = 0
        self.Tcell_IDs = []
        self.DC_IDs = []
        self.TaAPC_IDs = []
        self.agents = {} # agent data is stored in a dictionary, where keys are agent index/identifier numbers and each key's associated value is the agent.
        self.comps= {} #compartment data is stored in a dictionary, where keys are compartment index. Only a knee, just 0
        self.agents_removed = {} #Is this here for future building in of T cell egress? If so, I think we will need to build this in.
        self.agents_egressed = {}
        self.recorded_data = {'n_Tcell':0,          # total number of T cells
                              'n_Tcell_alive':0,    # total number of alive T cells
                              'n_Tcell_dead':0,
                              'n_DC_contact':0,     # number of first contacts with DC
                              'n_Tcell_contact':0,  # number of collisions between T cells
                              'n_TaAPC_contact':0,  # number of first contacts with TaAPC
                              'n_Tcell_active':0}   # number of activated T cells  
        
        self.cogT_goneDs = []
        
        self.antigen = 0 #antigen concentration in the LN used to determine probablity of DC -> cognate DC
        self.drain_rate = -1.3*10**-5 #[1/s] rate of antigen drainage out of the LN based on an assumed first order kinetics w/ 15 hr half life (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1457277/?page=4)
        self.drain_pct = 0.01 #fraction of antigen that actually makes it to the LN
        self.antigen_conc_0 = antigen_conc_0
        
        #Internal parameters that we are keeping track of in relation to simulation operation. Mostly for troubleshooting.
        self.interactions = 0
        self.moves = 0
        self.proliferations = 0
        self.bound_updates = 0
        self.data_records = 0
        
    def distance(self, ID1, ID2):
        '''
        Given the IDs of two agents, return the distance between their centers (positions).

        Parameters
        ----------
        ID1 : double
            ID of first agent.
        ID2 : double
            ID of second agent.

        Returns
        -------
        None.

        '''
        ag1 = self.agents[ID1]
        ag2 = self.agents[ID2]
        
        return np.sqrt((ag1.pos[0] - ag2.pos[0])*(ag1.pos[0] - ag2.pos[0])
                       + (ag1.pos[1] - ag2.pos[1])*(ag1.pos[1] - ag2.pos[1]) 
                       + (ag1.pos[2] - ag2.pos[2])*(ag1.pos[2] - ag2.pos[2]))


    def get_Tcell_ID(self):
        Tcell_ID = []
        cond = (ID for ID in self.agents if isinstance(self.agents[ID], Tcell)) # if T cell   
        for ID in cond:
            Tcell_ID.append(ID)
        
        return Tcell_ID

    def get_Tcell_dead_ID(self):
        Tcell_ID = []
        cond = (ID for ID in self.agents_removed if isinstance(self.agents_removed[ID], Tcell)) # if T cell   
        for ID in cond:
            Tcell_ID.append(ID)
        
        return Tcell_ID

    def get_DC_ID(self):
        DC_ID = []
        cond = (ID for ID in self.agents if isinstance(self.agents[ID], DC))        
        for ID in cond:
            DC_ID.append(ID)
        
        return DC_ID
    
    def get_TaAPC_ID(self):
        TaAPC_ID = []
        cond = (ID for ID in self.agents if isinstance(self.agents[ID], TaAPC))        
        for ID in cond:
            TaAPC_ID.append(ID)
        
        return TaAPC_ID

    def get_cognate_ID(self):
        cognate_ID = []
        cond = (ID for ID in self.agents if self.agents[ID].cognate)        
        for ID in cond:
            cognate_ID.append(ID)
        
        return cognate_ID
        
    def check_overlap(self, ID):
        '''
        Checks whether input agent (as inputted by ID number) overlaps with any other agent.

        Parameters
        ----------
        ID : int
            Input agent to check positional overlap with existing agents.

        Returns
        -------
        int
            ID of overlapping agent if overlap exists.
            0 if no overlaps exist

        '''
        
        ag = self.agents[ID]
        
        # Looping condition = IDs to test against input ID, not inclusive of ID itself.
        cond = (ID_test for ID_test in self.agents if ID_test != ID)
        for ID_test in cond:
            ag_test = self.agents[ID_test]
            
            # if distance is less than sum of radii then return True.
            if self.distance(ID, ID_test) < (ag.radius + ag_test.radius + self.error):
                return ID_test
            
        return 0

    def check_overlap_DC(self, ID):
        '''
        Checks whether input T cell agent (as inputted by ID number) overlaps with any DC agent. Note that this function does not check for T cells vs T cells.

        Parameters
        ----------
        ID : int
            Input agent to check positional overlap with existing agents.

        Returns
        -------
        int
            ID of overlapping agent if overlap exists.
            0 if no overlaps exist

        '''
        
        ag = self.agents[ID]
        
        # Looping condition = IDs to test against input ID, not inclusive of ID itself.
        cond = (ID_test for ID_test in self.get_DC_ID())
        for ID_test in cond:
            ag_test = self.agents[ID_test]
            
            # if distance is less than sum of radii then return True.
            if self.distance(ID, ID_test) < (ag.radius + ag_test.radius + self.error):
                return ID_test
            
        return 0

    def correct_pos(self, ID):
        '''
        Corrects position of input agent (as inputted by ID number) if overlaps exist.

        Parameters
        ----------
        ID : int
            ID # of the given agent.

        Returns
        -------
        None.

        '''
        ag = self.agents[ID]
        
        while self.check_overlap(ID) > 0:
            ID_test = self.check_overlap(ID)
            if not isinstance(ID_test, DC):
                break
            ag_t = self.agents[ID_test]
            
            a = ag_t.radius + ag.radius
            b = self.distance(ID, ID_test)
            
            # Express lines connecting points as vectors:
            dist_vec = ag.pos - ag_t.pos
            
            # Express direction of this agent as a vector by converting from spherical vector to rectangular:
            x = np.sin(ag.phi) * np.cos(ag.theta)
            y = np.sin(ag.phi) * np.sin(ag.theta)
            z = np.cos(ag.phi)
            
            ag_direction = np.array([x, y, z])
            
            ang_A = np.arccos(np.clip(np.dot(dist_vec, ag_direction) / (np.linalg.norm(ag_direction) * np.linalg.norm(dist_vec)), -1, 1))
            
            # Compute using law of sines:
            ang_B = np.arcsin(np.sin(ang_A) * b/a)
            ang_C = np.pi - ang_A - ang_B
            
            c = a * np.sin(ang_C) / np.sin(ang_A)
            
            xx = c * np.sin(ag.phi) * np.cos(ag.theta) + self.error/2
            yy = c * np.sin(ag.phi) * np.sin(ag.theta) + self.error/2
            zz = c * np.cos(ag.phi) + self.error/2
            shift = np.array([xx, yy, zz])
            
            self.agents[ID].pos -= shift


    def bound_correct(self, ID):
        '''
        Given the ID # of a given agent, correct its location based on boundary conditions. 
        When a cell completely leaves the computational domain, it is regenerated just outside (tangent to interface) at the face of the opposite exit face. 
        If cell leaves out lymphatic vessel patches as defined on simulation space, it will be removed from active agent directory.

        Parameters
        ----------
        ID : int
            ID # of the given agent.

        Returns
        -------
        None.

        '''
        ag = self.agents[ID]
        [x,y,z] = ag.pos*6
        dist = np.sqrt(np.sum(((ag.pos - self.border)*6) **2))
        
        if dist - self.bounds > 6:
            self.all_pos[int(ag.pos[0]),int(ag.pos[1]),int(ag.pos[2])] = 1
            if (z >= 0.9*self.bounds or z <= 0.9*self.bounds*-1) and dist >= self.bounds + 11.66 and ag.cognate and self.time > 500*60*60:
                print('My position is ' + str(ag.pos))
                print('I left at ' + str(self.time))
                print('The bounds are at ' + str(self.bounds))
                print('My distance is ' + str(dist))
                self.agents_egressed[ID] = self.agents[ID]
                del self.agents[ID]
                self.goneT += 1
                
            else:
                edge = self.bounds*(ag.pos-self.border)*6/(np.sqrt(sum(((ag.pos-self.border)*6)**2))) #vector representing the edge of the simulation space in the given direction.
                self.agents[ID].pos = np.round(((ag.pos - self.border)*6 - 2*edge)/6 + self.border) #correction shifting the cell back into the simulation space.
                self.all_pos[int(self.agents[ID].pos[0]), int(self.agents[ID].pos[1]), int(self.agents[ID].pos[2])] = 0

                


    def Initiate(self):
        '''
        Initiate a simulation using parameters defined in this file.
        This includes defining LN space size and boundaries, creating agent repetoire, generating initial positions, etc.

        Returns
        -------
        None.

        '''
        # Define size and boundaries of LN space:
        Tcell_vol = 4/3*(6/2)**3 # total volume occupied by T cell
        
        Tcell_n = self.agent_ct['Tcell']
        DC_n = self.agent_ct['DC']
        LN_vol = (Tcell_n*Tcell_vol + DC_n* Tcell_vol*7)/self.density_Tcell
        self.bounds = (3/4*LN_vol)**(1/3)
        self.bounds_int = int(np.ceil(self.bounds/6))
        self.border = int(np.ceil(self.bounds_int*10))
        
        self.all_pos = np.ones((self.border * 2,self.border * 2,self.border * 2))
        self.one_soma = self.soma_lattice()
        self.all_soma = np.zeros((self.border * 2,self.border * 2,self.border * 2))
        self.all_taapc = np.zeros((self.border * 2, self.border * 2, self.border * 2))
        
        # Record the simulation start time
        self.start_time = time.time()
        
        # Initialize the Knee drainage compartment
        self.comps['Knee'] = Knee(self.antigen_conc_0)
        
        rand.seed(self.seed)

        ID_run = 0
        # Add defined number of each type of agent to agents:
        for i in range(self.agent_ct['Tcell']):
            self.agents[ID_run] = Tcell()
            self.agents[ID_run].set_avidity(s = self.affinity_scale, seed = rand.randint(0,1000000))
            self.Tcell_IDs.append(ID_run)
            ID_run += 1
        for i in range(self.agent_ct['DC']):
            self.agents[ID_run] = DC()
            self.DC_IDs.append(ID_run)
            ID_run += 1
        for i in range(self.agent_ct['TaAPC']):
            self.agents[ID_run] = TaAPC()
            self.TaAPC_IDs.append(ID_run)
            ID_run += 1
            
        # Generate initial positions of T cell agents:
        temp_all_pos = np.copy(self.all_pos)
        for ID in self.Tcell_IDs:
            ag = self.agents[ID]
            
            phi = np.random.rand()*np.pi
            theta = np.random.rand()*2*np.pi
            r = np.random.rand()*self.bounds
            
            x = np.round(r*np.sin(phi)*np.cos(theta)/6 + self.border)
            y = np.round(r* np.sin(phi)*np.sin(theta)/6 + self.border)
            z = np.round(r*np.cos(phi)/6 + self.border)
            
            self.agents[ID].pos = np.array([x,y,z])
            temp_all_pos[int(x),int(y),int(z)] = 0
        self.all_pos = np.copy(temp_all_pos)
        
        #place all TaAPCs
        temp_taapc_pos = np.copy(self.all_taapc)
        for ID in self.TaAPC_IDs:
            ag = self.agents[ID]
            
            phi = np.random.rand()*np.pi
            theta = np.random.rand()*2*np.pi
            r = np.random.rand()*self.bounds
            
            x = np.round(r*np.sin(phi)*np.cos(theta)/6 + self.border)
            y = np.round(r* np.sin(phi)*np.sin(theta)/6 + self.border)
            z = np.round(r*np.cos(phi)/6 + self.border)
            
            ag.pos = np.array([x,y,z])
            temp_taapc_pos[int(x),int(y),int(z)] = 1
        
        self.all_taapc = np.copy(temp_taapc_pos)
        
            # while self.check_overlap(ID):
            #     phi = np.random.rand()*np.pi
            #     theta = np.random.rand()*2*np.pi
            #     r = np.random.rand()*self.bounds
            
            #     x = r*np.sin(phi)*np.cos(theta)
            #     y = r* np.sin(phi)*np.sin(theta)
            #     z = r*np.cos(phi)
            
            #     self.agents[ID].pos = np.array([x,y,z])
            
        # for ID in self.DC_IDs:
        #     ag = self.agents[ID]
            
        #     phi = np.random.rand()*np.pi
        #     theta = np.random.rand()*2*np.pi
        #     r = np.random.rand()*self.bounds
            
        #     x = r*np.sin(phi)*np.cos(theta)
        #     y = r* np.sin(phi)*np.sin(theta)
        #     z = r*np.cos(phi)
            
        #     self.agents[i].pos = np.array([x,y,z])
        #     while self.check_overlap(ID):
        #         phi = np.random.rand()*np.pi
        #         theta = np.random.rand()*2*np.pi
        #         r = np.random.rand()*self.bounds
            
        #         x = r*np.sin(phi)*np.cos(theta)
        #         y = r* np.sin(phi)*np.sin(theta)
        #         z = r*np.cos(phi)
            
        #         self.agents[ID].pos = np.array([x,y,z])
                
        # for ID in self.TaAPC_IDs:
        #     ag = self.agents[ID]
            
        #     phi = np.random.rand()*np.pi
        #     theta = np.random.rand()*2*np.pi
        #     r = np.random.rand()*self.bounds
            
        #     x = r*np.sin(phi)*np.cos(theta)
        #     y = r* np.sin(phi)*np.sin(theta)
        #     z = r*np.cos(phi)
            
        #     self.agents[i].pos = np.array([x,y,z])
        #     while self.check_overlap(ID):
        #         phi = np.random.rand()*np.pi
        #         theta = np.random.rand()*2*np.pi
        #         r = np.random.rand()*self.bounds
            
        #         x = r*np.sin(phi)*np.cos(theta)
        #         y = r* np.sin(phi)*np.sin(theta)
        #         z = r*np.cos(phi)
            
        #         self.agents[ID].pos = np.array([x,y,z])

        
        # Set cognate T cells and cognate DCs
        # cognate_Tcell_ct = 0;
        # for i in self.Tcell_IDs:
        #     if cognate_Tcell_ct < self.cognate_Tcell_n:
        #         self.agents[i].cognate = True
        #         cognate_Tcell_ct += 1
                
        # cognate_DC_ct = 0;
        # for i in self.DC_IDs:
        #     if cognate_DC_ct < self.cognate_DC_n:
        #         self.agents[i].cognate = True
        #         cognate_DC_ct += 1
        
        self.ID_max = ID_run
        
        print('I have begun!')
        for ID in self.get_Tcell_ID():
            #print(self.agents[ID].div_can)
            #print(self.agents[ID].div_ability)
            #print(self.agents[ID].S)
            print(self.agents[ID].avidity)
            
        self.aff_plot()
        
        
    def Move(self):
        
        for ID in self.get_Tcell_ID():
            T = self.agents[ID]
            if T.walk:
                # print(str(ID) + " just took a step")
                x = int(T.pos[0])
                y = int(T.pos[1])
                z = int(T.pos[2])
                self.all_pos[x-2:x + 3, y - 2:y + 3, z - 2:z + 3] = T.move_T(self.all_pos)
                self.bound_correct(ID)
        
        self.moves += 1
        #print('I have moved' + str(self.moves) + 'times')

    def interact_DC(self, ID1):
        
        #Check if current T cell is cognate
        is_cog = self.agents[ID1].cognate
        # if already interacting, update parameters:
            
        
        if self.agents[ID1].contact_status and self.agents[ID1].contact_can: 
            #If we are 
            self.agents[ID1].contact_t += self.tau

            # if maximum interaction time is reached:
            if self.agents[ID1].contact_t > self.agents[ID1].contact_tlim:
                self.agents[ID1].contact_t = 0
                self.agents[ID1].contact_status = False
                self.agents[ID1].contact_can = False
                self.agents[ID1].walk = True
                self.T_to_DC -= 1
                
        else: # else start interactions
            if self.agents[ID1].contact_can: # if able to interact
                self.T_to_DC += 1
                self.agents[ID1].contact_t += self.tau
                self.agents[ID1].contact_status = True
                self.agents[ID1].walk = False
                
                
                if is_cog:
                    #print('My stimulation level is ' + str(self.agents[ID1].S))
                    self.agents[ID1].stimulate_T()
                    self.stimcount += 1
                    #print('I am stimulating again, I have stimulated ' + str(self.stimcount) + ' times')
                    #print('My new stimulation level is ' + str(self.agents[ID1].S))
                    
                    if self.agents[ID1].S > 180:
                        self.agents[ID1].can_inhib = True
                        #print('I am cell ' + str(ID1) + ' and my stimluation level is ' + str(self.agents[ID1].S) + ' and my can_inhib value is ' + str(self.agents[ID1].can_inhib))
                        
                
                # Set time limit for interaction 
                    if self.agents[ID1].S < 100:
                        mean = self.agents[ID1].tlim_p1_cog[0]
                        sd = self.agents[ID1].tlim_p1_cog[1]
                        self.agents[ID1].contact_tlim = np.random.normal(mean, sd)
                            
                    elif (self.agents[ID1].S >= 100 and self.agents[ID1].S < 200):
                        mean = self.agents[ID1].tlim_p2_cog[0]
                        sd = self.agents[ID1].tlim_p2_cog[1]
                        self.agents[ID1].contact_tlim = np.random.normal(mean, sd)
                        if self.agents[ID1].mature == True:
                            if self.agents[ID1].div_can:
                                #if self.div_check_temp:
                                    #print(self.time)
                                    #print(self.agents[ID1].S)
                                self.agents[ID1].div_ability = True
    
                    elif self.agents[ID1].S >= 200:
                        mean = self.agents[ID1].tlim_p3_cog[0]
                        sd = self.agents[ID1].tlim_p3_cog[1]
                        self.agents[ID1].contact_tlim = np.random.normal(mean, sd)
                        if self.agents[ID1].S >= 300:
                            if self.agents[ID1].div_can:                        
                                self.agents[ID1].div_ability = True
                                #if self.div_check_temp:
                                    #print(self.time)
                                    #print(ID1)
                                    #print(self.agents[ID1].S)
                                    
                            self.agents[ID1].mature = True
                            print(str(ID1) + ' is ready to proliferate.')
                elif not is_cog:
                    mean = self.agents[ID1].tlim_p1_noncog[0]
                    sd = self.agents[ID1].tlim_p1_noncog[1]
                    self.agents[ID1].contact_tlim = np.random.normal(mean, sd)
    
    def interact_TaAPC(self, ID1):
        
        ag = self.agents[ID1]
            
        if not ag.inhib:
            print('' + str(ag.inhib))
            if ag.can_inhib:
                print(' ' + str(ag.can_inhib))
                ag.pd1_inhib()
                print('Im inhibited and Im cell ' + str(ID1) + '. My stimulation level is ' + str(ag.S))
            

    def Interact(self):
        '''
        Initiate interactions among all agents in contact with one another.

        Returns
        -------
        None.

        '''
        #Reset counting stats to 0 for the timestep
        self.T_to_DC = 0
        self.T_to_TaAPC = 0
        self.cogDCs = 0
        
        #update antigen values in various compartments
        self.comps['Knee'].inflam(self.tau) #update antigen values in Knee
        self.comps['Knee'].update_cells(self.kneeT) #Drain T cells to Knee

        #Check for T cells interacting with Soma:
        for ID in self.get_Tcell_ID():
            [x,y,z] = self.agents[ID].pos
            if self.all_soma[int(x),int(y),int(z)] > 0:
                if self.agents[ID].alive:
                    self.interact_DC(ID)
            
            if self.all_taapc[int(x),int(y),int(z)] > 0:
                if self.agents[ID].alive:
                    #print('I am cell ' + str(ID) + ' and I interacted with a TaAPC at ' + str(self.time))
                    #print('My can_inhib value is ' + str(self.agents[ID].can_inhib))
                    self.interact_TaAPC(ID)
                
                # elif self.all_TaAPC > 0:
                #     self.interact_TaAPC()
                #     # Kill cell if interacting with TaAPC
                #     self.T_to_TaAPC += 1
        
        self.interactions += 1
        #print('I have interacted' + str(self.interactions) + 'times')

    def generate_Tcell(self, ID, prolif = True):
        '''
        Generate/"Proliferate" a T cell.

        Parameters
        ----------
        ID : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        # Generate random new unoccupied position adjacent to T cell:
        new_Tcell = Tcell()
        new_Tcell.cognate = self.agents[ID].cognate
        new_Tcell.div_can = False
        new_Tcell.div_ability = False
        
        
        check = True

        # Random position of new T cell:
        x = np.random.randint(-1,2)
        y = np.random.randint(-1,2)
        z = np.random.randint(-1,2)
        shift = np.array([x, y, z])

        new_Tcell.pos = self.agents[ID].pos + shift
        if prolif:
            new_Tcell.avidity = self.agents[ID].avidity
            new_Tcell.can_inhib = self.agents[ID].can_inhib
        
        else:
            new_Tcell.set_avidity(s = self.affinity_scale)
        
        new_Tcell.S = self.agents[ID].S/2
        
        self.liveT += 1
        
        self.ID_max += 1
        
        new_ID = self.ID_max
        
        self.agents[new_ID] = new_Tcell
        self.bound_correct(new_ID)
            

    def prolif(self):
        for ID in self.get_Tcell_ID():
            #print(str(ID))
            if self.agents[ID].div_can:
                #print('div can yes')
                if self.agents[ID].div_ability: # if can divide:
                    #print('div ability yes')
                    if self.agents[ID].div_status: # if currently dividing:
                        #print('dividing, gimme a minute')
                        self.agents[ID].div_t += self.tau
                        if self.agents[ID].div_t >= self.agents[ID].div_tlim: # if able to split
                            #print('Im ready to split')
                            self.generate_Tcell(ID) # generate T cell next to this one
                            #print('I have created a new T cell')
                            
                            #print('resetting division parameters')
                            self.agents[ID].div_status = False
                            self.agents[ID].div_ability = False
                            self.agents[ID].div_can = False
                            self.agents[ID].div_t = 0
                            self.agents[ID].S = self.agents[ID].S/2
                            #print('division params reset')
                            self.proliferations += 1
                            print('I have bow chika wow wowed' + str(self.proliferations) + 'times')
                            
                            
                            if self.agents[ID].div_n >= self.agents[ID].div_nlim:
                                self.agents[ID].ability = False
                            
                    else: # if not currently dividing, begin dividing:
                        self.agents[ID].div_status = True
                            
                        self.agents[ID].div_n += 1
            else: # if cannot divide due to cooldown period
                self.agents[ID].div_can_cd -= self.tau
                
                if self.agents[ID].div_can_cd <= 0:
                    self.agents[ID].div_can_cd = self.agents[ID].div_can_cdlim
                    self.agents[ID].div_can = True
        
        
        

    def update_bounds(self):
        '''
        Updates (increases and decreases) size of simulated space to maintain overall T cell density.

        Returns
        -------
        None.

        '''
        # Define size and boundaries of LN space:
        Tcell_vol = 4/3*(6/2)**3 # total volume occupied by T cell
        
        Tcell_n = len(self.get_Tcell_ID())
        DC_n = len(self.get_DC_ID())
        LN_vol = (Tcell_n*Tcell_vol + DC_n* Tcell_vol*7)/self.density_Tcell
        self.bounds = (3/4*LN_vol)**(1/3)
        temp_shift = np.ceil(self.bounds/6) - self.bounds_int 
        self.bounds_int = np.ceil(self.bounds/6)
        #print(self.bounds_int)
        # self.border = int(np.ceil(self.bounds_int*10))
        
        #print(temp_shift)
        
        # if temp_shift > 0:
        #     new_pos = np.ones(shape = (self.border*2, self.border*2, self.border*2))
            
        #     print(np.shape(new_pos))
        #     new_pos[:np.shape(self.all_pos)[0], :np.shape(self.all_pos)[1], :np.shape(self.all_pos)[2]] = self.all_pos
        #     self.all_pos = new_pos
        
        # self.bound_updates += 1
        #print('I have updated my bounds' + str(self.bound_updates) + 'times')


    def update_time(self):
        # Update simulation time.
        self.time += self.tau
        self.run_pct = self.time/self.end*100
        
        #Reset dead cell counter for the timestep
        self.deadT = 0
        self.goneT = 0
        self.kneeT = 0
        
        for ID in self.get_Tcell_ID(): # for T cells
            self.agents[ID].decay_T()
            
            if self.agents[ID].inhib:
                self.agents[ID].inhib_t -= self.tau
                
                if self.agents[ID].inhib_t <= 0:
                    self.agents[ID].inhib = False
                    
            if self.agents[ID].lifespan <= self.agents[ID].age:
                self.agents[ID].alive = False
            if self.agents[ID].alive: # update age if alive
                self.agents[ID].age += self.tau
                
                # Set T cell behavior "phase" based on age
                # if self.agents[ID].age < 60*60*8:
                #     self.agents[ID].phase = 1
                # elif self.agents[ID].age < 60*60*24:
                #     self.agents[ID].phase = 2
                # else:
                #     self.agents[ID].phase = 3
                
                if not self.agents[ID].contact_can:
                    self.agents[ID].contact_can_cd -= self.tau
                    if self.agents[ID].contact_can_cd <= 0:
                        self.agents[ID].contact_can_cd = self.agents[ID].contact_can_cdlim
                        self.agents[ID].contact_can = True
                        
            else: # update how long cell has been dead for if dead
                self.agents[ID].death_t += self.tau
                if self.agents[ID].death_t > self.agents[ID].death_tlim: # clear cell if dead long enough
                    self.agents_removed[ID] = self.agents[ID]
                    self.deadT += 1 # Update dead cell counter
                    del self.agents[ID]
                    
            if rand.random() <= self.Tegress:    
                #print(str(ID) + ' left at ' + str(self.time))
                self.agents_egressed[ID] = self.agents[ID]
                if self.agents[ID].mature:
                    self.kneeT += 1
                del self.agents[ID]
                self.goneT += 1
                
            if rand.random() <= self.Tegress:
                self.generate_Tcell(ID, prolif = False)
                    
        for ID in self.get_DC_ID():
            DC = self.agents[ID]
            if DC.is_alive:
                DC.lifespan -= self.tau
                if DC.lifespan <=0:
                    [x,y,z]  = self.agents[ID].pos
                    self.all_soma[(x - 3):(x + 4), (y - 3):(y + 4), (z - 3):(z + 4)] -= np.copy(self.one_soma)
                    del self.agents[ID]
                    print('##################### I HAVE DIED #####################')
                    print('##################### WOE IS ME!!!#####################')
                    self.cogDCs -= 1

    def vis(self, save_file = False, filename = 'LN_Visualization'):
        '''
        Plot and visualize all agents within the computational domain.

        Returns
        -------
        None.

        '''
        fig = plt.figure(figsize=(12,12), dpi=self.vis_res)
        ax = fig.add_subplot(111, projection='3d')

        # Setting axes properties:
        xBound = self.bounds
        yBound = self.bounds
        zBound = self.bounds

        ax.set_xlim3d([-xBound, xBound])
        ax.set_xlabel('X')

        ax.set_ylim3d([-yBound, yBound])
        ax.set_ylabel('Y')

        ax.set_zlim3d([-zBound, zBound])
        ax.set_zlabel('Z')

        ax.view_init(None, 45)
        
        #Set colors for different avidities
        
        cmap_f = cm.ScalarMappable(cmap = 'rainbow')
        cmap_f.set_clim(0,3.5)

        for ID in self.get_Tcell_ID():
            # print("I'm plotting agent number" + str(ID))
            if not self.agents[ID].alive:
                c = self.agents[ID].color_dead
            elif self.agents[ID].cognate:
                c = cmap_f.to_rgba(self.agents[ID].avidity)
            else:
                c = self.agents[ID].color_noncog
                
            plotSph(ax, disk(self.agents[ID].pos*6-self.border*6, self.agents[ID].radius), c)

        for ID in self.get_DC_ID():
            if not self.agents[ID].cognate:
                plotSph(ax, disk(self.agents[ID].pos*6-self.border*6, self.agents[ID].radius_soma), self.agents[ID].color_act)
                plotSph(ax, disk(self.agents[ID].pos*6-self.border*6, self.agents[ID].radius), self.agents[ID].color)
            else:
                plotSph(ax, disk(self.agents[ID].pos*6-self.border*6, self.agents[ID].radius_soma), self.agents[ID].color_act_cog)
                plotSph(ax, disk(self.agents[ID].pos*6-self.border*6, self.agents[ID].radius), self.agents[ID].color_cog)
            
        for ID in self.get_TaAPC_ID():
            plotSph(ax, disk(self.agents[ID].pos*6-self.border*6, self.agents[ID].radius), self.agents[ID].color)
            
        
        if save_file == False:
            
            plt.show()
            plt.clf()
            
        elif save_file == True:
            filename = filename + '.png'
            plt.savefig(filename)
            plt.clf()

    
    def compile_data(self):
        for ID in self.get_Tcell_ID():
            if self.agents[ID].alive:
                self.recorded_data['n_Tcell_alive'] += 1
            else:
                self.recorded_data['n_Tcell_dead'] += 1
                
            self.recorded_data['n_DC_contact'] += self.agents[ID].contact_DC
            self.recorded_data['n_Tcell_contact'] += self.agents[ID].contact_Tcell
            self.recorded_data['n_TaAPC_contact'] += self.agents[ID].contact_TaAPC
            
            if self.agents[ID].div_ability or self.agents[ID].div_n >= 8:
                self.recorded_data['n_Tcell_active'] += 1

        for ID in self.get_Tcell_dead_ID():
            self.recorded_data['n_Tcell_dead'] += 1
        
        self.recorded_data['n_Tcell'] = self.recorded_data['n_Tcell_alive'] + self.recorded_data['n_Tcell_dead']
        
        
    def export_data(self):
        self.liveT = self.liveT - self.deadT - self.goneT
        
        self.data_records += 1
        
        #print('I have recorded data' +str(self.data_records) + 'times.')
        
        if self.chkpt25 and self.run_pct >= 25:
            self.chkpt25 = False
            print('The simulation is at 25%, and it has been running for ' + str((time.time() - self.start_time)/60) + 'minutes and it is ' + str(time.time()))
            #self.vis()
            
        elif self.chkpt50 and self.run_pct >= 50:
            self.chkpt50 = False
            print('The simulation is at 50%, and it has been running for ' + str((time.time() - self.start_time)/60) + 'minutes and it is ' + str(time.time()))
            #self.vis()
            #self.aff_plot()
        
        elif self.chkpt75 and self.run_pct >= 75:
            self.chkpt75 = False
            print('The simulation is at 75%, and it has been running for ' + str((time.time() - self.start_time)/60) + 'minutes and it is ' + str(time.time()))
            #self.vis()
            
        elif self.chkpt100 and self.run_pct >= 100:
            self.chkpt100 = False
            print('The simulation is at 100%, and it has been running for ' + str((time.time() - self.start_time)/60) + 'minutes and it is ' + str(time.time()))
            #self.vis()
            #self.aff_plot()
            total_runtime = str((time.time() - self.start_time)/60)
            with open('run_log.txt', 'w') as f:
                f.write('Runtime: ' + total_runtime)
                f.write('\n')
                f.write('Tcell Start Count: ' + str(self.agent_ct['Tcell']))
                f.write('\n')
                f.write('Affinity Scale: ' + str(self.affinity_scale))
                f.write('\n')
                f.write('Affinity Shape: ' + str(self.affinity_shape))
                f.write('\n')
                f.write('Cog T Start Fraction: ' + str(self.cognate_Tcell))
                f.write('\n')
                f.write('Simulation Time: ' + str(self.time))
                f.write('\n' + 'Number of Timesteps: ' + str(self.time/self.tau))
                f.write('\n' + 'Initial antigen concentration: ' + str(self.antigen_conc_0))
        
        avs= np.zeros(self.liveT)
        i = 0
        for ID in self.get_Tcell_ID(): # for T cells
            avs[i] = self.agents[ID].avidity
            i += 1
            
        av_avs = np.mean(avs)
        sd_avs = np.std(avs)
            
                
        
        return[self.liveT, self.goneT, self.comps['Knee'].antigen[self.comps['Knee'].intime - 1], self.comps['Knee'].Tcell[self.comps['Knee'].intime - 1], self.cogDCs, self.antigen, self.bounds, av_avs, sd_avs]
        
    def export_av_bins(self):
        bin_vals = np.linspace(0,6,31)
        avs = np.zeros(self.liveT)
        i = 0
        for ID in self.get_Tcell_ID():
            avs[i] = self.agents[ID].avidity
            i += 1
        avs_bins = plt.hist(avs, bin_vals)[0]
        
        return avs_bins
        
    def get_antigen(self, source_a):
         self.antigen = self.antigen + self.antigen*self.drain_rate*self.tau + source_a*3.2*10**-5*self.tau*self.drain_pct
         
    def DC_recruit(self):
        choose = False
        while not choose:
            phi = np.random.rand()*np.pi
            theta = np.random.rand()*2*np.pi
            r = np.random.rand()*self.bounds
            
            x = int(np.round(r*np.sin(phi)*np.cos(theta)/6 + self.border))
            y = int(np.round(r* np.sin(phi)*np.sin(theta)/6 + self.border))
            z = int(np.round(r*np.cos(phi)/6 + self.border))
            
            if self.all_pos[x,y,z] == 1:
                choose = True
                self.all_pos[x,y,z] = 0
                self.all_soma[(x - 3):(x + 4), (y - 3):(y + 4), (z - 3):(z + 4)] += np.copy(self.one_soma)
                #This currently has two weird things. One is that edge effects will mess this up slighlty. Two is that overlaping Somas will actually delete some of the other soma because of the zeros in the matrix.
                return np.array([x,y,z])
            
        print("Error in DC_recruit")
            
    def soma_lattice(self):
        D = int(42/6) #Diameter of Soma in Voxels
        lattice = np.ones((D,D,D))
        for x in range(D):
            for y in range(D):
                for z in range(D):
                    dist = (x-3)**2 + (y-3)**2 + (z-3)**2
                    if dist >= D/2:
                        lattice[x,y,z] = 0
        return lattice
    
    def Recruitment(self):
        is_recruit = self.comps['Knee'].DC_drain(self.tau)
        if is_recruit and len(self.get_DC_ID())/len(self.get_Tcell_ID()) < 1/156:
            print("I added a DC")
            new_DC = DC()
            new_DC.pos = self.DC_recruit()
            new_ID = self.ID_max
            self.cogDCs += 1
            self.ID_max += 1
            self.agents[new_ID] = new_DC
    
    def aff_plot(self, save_file = False, filename = 'Affinity_Histogram'):
        fig = plt.figure()
        affs = []
        for ID in self.get_Tcell_ID():
            affs.append(self.agents[ID].avidity)
        bin_size = int(len(affs)/4)
        plt.hist(affs, bins = 20)
        
        if save_file == False:
            
            plt.show()
            plt.clf()
            
        elif save_file == True:
            filename = filename + '.png'
            plt.savefig(filename)
            plt.clf()
            
        

def disk(pos, radius):
    '''
    Function generates data used for sphere plotting in pltSph function.

    Parameters
    ----------
    pos : 3x1 float np.array
        Coordinates of agent center.
    radius : float
        Radius of agent.

    Returns
    -------
    float list
        x-coordinates.
    float list
        y-coordinates.
    float list
        z-coordinates.

    '''
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    return x+pos[0], y+pos[1], z+pos[2]

def plotSph(ax, data, c):
    '''
    Sphere plotting in Matplotlib.

    Parameters
    ----------
    ax : figure subplot
        Subplot in which agent is to be plotted on.
    data : list of coordinate lists
        x, y, and z coordinates as transformed by disk function.
    c : color code
        Sphere color.

    Returns
    -------
    None.

    '''
    x, y, z = data[0], data[1], data[2]
    ax.plot_surface(x, y, z, rstride=4, cstride=4,
                    color = c, linewidth=0, alpha=0.5)

