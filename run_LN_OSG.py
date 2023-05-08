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
import argparse
import random

from Agent import Agent
from Tcell import Tcell
from DC import DC
from TaAPC import TaAPC
import LN


# Need to refine this
def sim(trials, agent_ct, cognate_Tcell, end = 60*60*24*8, vis_res = 40, affinity_scale = 1000, affinity_shape = 1.1, antigen_conc_0 = 10**3, seed = 0, fname = 'Figure'):
    
    col = ['n_Tcell, n_Tcell_alive, n_Tcell_dead, n_DC_contact, n_Tcell_contact, n_TaAPC_contact, n_Tcell_active']
    df = pd.DataFrame(columns=col)
    #Seed test
    
    for i in range(trials):
        LN1 = LN.LN(agent_ct, cognate_Tcell, end, vis_res, affinity_scale, antigen_conc_0 = antigen_conc_0, seed = seed)
        
        LN1.Initiate()
        steps = int(end/LN1.tau)
        if i == 0:
            dataBloc = np.zeros((trials, 9, steps))
            avs = np.zeros((trials, 30, 5))
            
        #LN1.vis(save_file = True, filename = "INIT_Vis_" + fname)
        #LN1.aff_plot(save_file = True, filename = "INIT_Affinity_" + fname)
        
        print("My affinity scale is " + str(LN1.affinity_scale))
        avs_ct = 0
        for j in range(steps):
            LN1.Recruitment()
            LN1.Move()
            LN1.Interact()
            LN1.prolif()
            LN1.update_bounds()
            LN1.update_time()
            dataBloc[i, :, j] = LN1.export_data()
            if j % (steps/4) == 0:
                avs[i, :, avs_ct] = LN1.export_av_bins()
                avs_ct += 1
            #LN1.vis()
        #LN1.vis(save_file = True, filename = "END_Vis_" + fname)
        #LN1.aff_plot(save_file = True, filename = "END_Affinity_" + fname)
        
        all_pos = LN1.all_pos
    
        
        data = []
        for key in LN1.recorded_data:
            data.append(LN1.recorded_data[key])
        
        df.append(data, ignore_index=True)
        
        t = list(range(len(dataBloc[0,0,:])))
        
        t = np.array(t)
        
        t = t*12/60/60
        
        ds = LN1.cogT_goneDs
        
    return [dataBloc, t, ds, avs]

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
            [dataBloc, t, ds] = sim(1, agent_ct, cognate_Tcell, end = time, vis_res = 200, affinity_mean = 1, affinity_shape = shape)
            
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

#%% Build Parser and determine if running from console or IPython
my_parser = argparse.ArgumentParser(fromfile_prefix_chars='@',description='Retreive input for model parameters')
my_parser.add_argument('SimTime',type=int, help = 'Number of simulated seconds')
my_parser.add_argument('CogT', type=float, help = 'Fraction of cognate T cells')
my_parser.add_argument('Tcell', type=int, help = 'Number of T cells to simulate')
my_parser.add_argument('scale', type = float, help = 'Scale for Maxwell distrobution')
my_parser.add_argument('shape', type = float, help = 'Shape factor for lognorm distro of affinities')
my_parser.add_argument('antigen', type = float, help = 'Initial antigen concentration in knee')
my_parser.add_argument('taapc', type = int, help = 'Number of TaAPCs inserted upon initialization')
my_parser.add_argument('seed', type = int, help = 'Seed for RNG.')

try:
    __IPYTHON__
except NameError:
    print("Not running in IPython so will parse arguments")
    args =  my_parser.parse_args()
    time = args.SimTime
    cognate_Tcell = args.CogT
    Tcell_num = args.Tcell
    scale = args.scale
    shape = args.shape
    a_conc_0 = args.antigen
    taapc = args.taapc
    seed = args.seed
    print('Parsed successfully.')
else:
    print("Running in IPython")
    print('Use defaults?')
    defaults = str(input())
    if defaults == 'n':
        conditions = [1.1, 2, 5, 10]
        print('Please input the simulation time in seconds')
        time = int(input())
        print('Please input the fraction of cognate T cells')
        cognate_Tcell = float(input())
        print('Please input the total number of T cells to simulate')
        Tcell_num = int(input())
        print('Please input the scale for the Maxwell distribution')
        #Maxwell Scale factor
        scale = float(input())
        print('Please input the shape factor for the Maxwell distribution')
        shape = float(input())
        print('Please input the inital antigen concentration in the knee')
        a_conc_0 = float(input())
        print('Please input the initil number of TaAPCs in the LN')
        taapc = int(input())
        print('What seed would you like to use?')
        seed = int(input())
    elif defaults == 'y':
        time = 864000/4*3
        cognate_Tcell = 1
        Tcell_num = 20
        scale = 1000
        shape = 1.1
        a_conc_0 = 10000
        taapc = 0
        seed = 0
    
agent_ct = {'Tcell':Tcell_num, 'DC':0, 'TaAPC':taapc}
filename = "Data_time" + str(time) + "_cog" + str(cognate_Tcell) + "_T" + str(Tcell_num) + "_scale" + str(scale) + "_shape" + str(shape) + ".csv"

[dataBloc, t, ds, avs] = sim(1, agent_ct, cognate_Tcell, end = time, vis_res = 200, affinity_scale = scale, affinity_shape = shape, antigen_conc_0 = a_conc_0, seed = seed, fname = filename)
#np.savetxt(fname = "dataBloc" + filename, X = dataBloc[0,:,:], delimiter=',')
#np.savetxt(fname = "avs" + filename, X = avs[0,:,:], delimiter=',')
ecount = [scale, a_conc_0, taapc]
ecount.append(dataBloc[0,0,-1])
np.savetxt(fname = "EndCounts" + filename, X = ecount, delimiter=',')

# plt.plot(t, dataBloc[0,0,:])
# plt.xlabel('Time (h)')
# plt.ylabel('T cells in LN')
# plt.show()

# plt.plot(t, dataBloc[0,2,:])
# plt.xlabel('Time (h)')
# plt.ylabel('Antigen Concentration in Knee')
# plt.show()























