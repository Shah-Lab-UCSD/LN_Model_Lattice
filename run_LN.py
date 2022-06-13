# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 15:42:11 2022

@author: jwjam
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

from Agent import Agent
from Tcell import Tcell
from DC import DC
from TaAPC import TaAPC
import LN


# Need to refine this
def sim(trials, agent_ct, cognate_Tcell, end = 60*60*24*8, vis_res = 40, affinity_mean = 1, affinity_shape = 1.1, antigen_conc_0 = 10**3):
    
    col = ['n_Tcell, n_Tcell_alive, n_Tcell_dead, n_DC_contact, n_Tcell_contact, n_TaAPC_contact, n_Tcell_active']
    df = pd.DataFrame(columns=col)
    
    for i in range(trials):
        LN1 = LN.LN(agent_ct, cognate_Tcell, end, vis_res)
        
        LN1.Initiate()
        steps = int(end/LN1.tau)
        if i == 0:
            dataBloc = np.zeros((trials, 7, steps))
            
        LN1.vis()
        
        for j in range(steps):
            LN1.Recruitment()
            LN1.Move()
            LN1.Interact()
            LN1.prolif()
            LN1.update_bounds()
            LN1.update_time()
            dataBloc[i, :, j] = LN1.export_data()
            #LN1.vis()
        LN1.vis()
        
        data = []
        for key in LN1.recorded_data:
            data.append(LN1.recorded_data[key])
        
        df.append(data, ignore_index=True)
        
        t = list(range(len(dataBloc[0,0,:])))
        
        t = np.array(t)
        
        t = t*12/60/60
        
        ds = LN1.cogT_goneDs
        
    return [dataBloc, t, ds]

def main():
    num_runs = 4
    conditions = [1.1, 2, 5, 10]
    time = 60*60*24*4
    agent_ct = {'Tcell':100, 'DC':0, 'TaAPC':0}
    cognate_Tcell = 1
    antigen_data = np.zeros((num_runs*len(conditions), int(time/12)))
    Tcell_data = np.zeros((num_runs*len(conditions), int(time/12)))
    run_count = 0
    for shape in conditions:
        for run in range(num_runs):
            [dataBloc, t, ds] = sim(1, agent_ct, cognate_Tcell, end = time, vis_res = 40, affinity_mean = 1, affinity_shape = shape)
            
            antigen_data[run_count,:] = dataBloc[0,2,:]
            Tcell_data[run_count,:] = dataBloc[0,0,:]
            
            plt.plot(t, dataBloc[0,0,:])
            plt.xlabel('Time (h)')
            plt.ylabel('T cells in LN')
            plt.show()
            
            plt.plot(t, dataBloc[0,2,:])
            plt.xlabel('Time (h)')
            plt.ylabel('Antigen Concentration in Knee')
            plt.show()

    return [Tcell_data, antigen_data]

def main2():
    conditions = [1.1, 2, 5, 10]
    print('Please input the simulation time in seconds')
    time = int(input())
    print('Please input the fraction of cognate T cells')
    cognate_Tcell = float(input())
    print('Please input the total number of T cells to simulate')
    Tcell = int(input())
    print('Please input the mean for the lognormal distribution')
    mean = float(input())
    print('Please input the shape factor for the lognormal distribution')
    shape = float(input())
    print('Please input the inital antigen concentration in the knee')
    a_conc_0 = float(input())
    agent_ct = {'Tcell':Tcell, 'DC':0, 'TaAPC':0}
    filename = "Data_time" + str(time) + "_cog" + str(cognate_Tcell) + "_T" + str(Tcell) + "_mean" + str(mean) + "_shape" + str(shape) + ".csv"
    
    [dataBloc, t, ds] = sim(1, agent_ct, cognate_Tcell, end = time, vis_res = 40, affinity_mean = mean, affinity_shape = shape, antigen_conc_0 = a_conc_0)
    np.savetxt(fname = filename, X = dataBloc[0,:,:], delimiter=',')
    
    plt.plot(t, dataBloc[0,0,:])
    plt.xlabel('Time (h)')
    plt.ylabel('T cells in LN')
    plt.show()
    
    plt.plot(t, dataBloc[0,2,:])
    plt.xlabel('Time (h)')
    plt.ylabel('Antigen Concentration in Knee')
    plt.show()
    
    return 0






















