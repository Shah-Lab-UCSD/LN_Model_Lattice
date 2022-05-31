# -*- coding: utf-8 -*-
"""
Created on Fri May 27 16:07:24 2022

@author: davem
"""
time = 60*60*24*10
cog_frac = 0.001

shape = 1.1
mean = 1
a_conc = 10**4


with open('run_conditions.txt','w') as f:
    
    for i in range(100):
        if i < 10:
            Tcell = 1000
            f.write(str(time) + ',' + str(cog_frac) + ',' + str(Tcell) + ',' + str(mean) + ',' + str(shape) + ',' + str(a_conc))
            f.write('\n')
        elif i < 20:
            Tcell = 2000
            f.write(str(time) + ',' + str(cog_frac) + ',' + str(Tcell) + ',' + str(mean) + ',' + str(shape) + ',' + str(a_conc))
            f.write('\n')
        elif i < 30:
            Tcell = 3000
            f.write(str(time) + ',' + str(cog_frac) + ',' + str(Tcell) + ',' + str(mean) + ',' + str(shape) + ',' + str(a_conc))
            f.write('\n')
        elif i < 40:
            Tcell = 4000
            f.write(str(time) + ',' + str(cog_frac) + ',' + str(Tcell) + ',' + str(mean) + ',' + str(shape) + ',' + str(a_conc))
            f.write('\n')
        elif i < 50:
            Tcell = 5000
            f.write(str(time) + ',' + str(cog_frac) + ',' + str(Tcell) + ',' + str(mean) + ',' + str(shape) + ',' + str(a_conc))
            f.write('\n')
        elif i < 60:
            Tcell = 6000
            f.write(str(time) + ',' + str(cog_frac) + ',' + str(Tcell) + ',' + str(mean) + ',' + str(shape) + ',' + str(a_conc))
            f.write('\n')
        elif i < 70:
            Tcell = 7000
            f.write(str(time) + ',' + str(cog_frac) + ',' + str(Tcell) + ',' + str(mean) + ',' + str(shape) + ',' + str(a_conc))
            f.write('\n')
        elif i < 80:
            Tcell = 10000
            f.write(str(time) + ',' + str(cog_frac) + ',' + str(Tcell) + ',' + str(mean) + ',' + str(shape) + ',' + str(a_conc))
            f.write('\n')
        elif i < 90:
            Tcell = 20000
            f.write(str(time) + ',' + str(cog_frac) + ',' + str(Tcell) + ',' + str(mean) + ',' + str(shape) + ',' + str(a_conc))
            f.write('\n')
        elif i < 100:
            Tcell =25000
            f.write(str(time) + ',' + str(cog_frac) + ',' + str(Tcell) + ',' + str(mean) + ',' + str(shape) + ',' + str(a_conc))
            f.write('\n')