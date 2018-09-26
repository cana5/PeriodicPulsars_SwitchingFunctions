# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 11:06:35 2018

@author: cana5
"""
# Import the following programs:
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# Define the following to convert values to seconds:
seconds_in_year = 86400 * 365
seconds_in_day = 86400

# Define the independent variable:
time = np.linspace(50000, 55000, 1000)
time_seconds = time * 86400

# Define the signal model: 
def SignalModel(time, nudot1, nudot2, nuddot, T, tAB, tBC, tCD, phi0, sigma):

    # This is the independent variable minus the MJD value:
    time = time - 49621 * 86400
    
    # This is the number of fine measurements:
    Nfine = 10000
    
    # This is the amount of smoothing
    DeltaT = 100.0*86400  
    
    # This is the fraction to shift by:
    frac_rec = 4  

    # Get the Perera switching of spindown:
    # Define an array:
    time_fine = np.linspace(time[0]-DeltaT/2., time[-1]+DeltaT/2., Nfine)
    
    # This gives an array of zeroes; add the constant 'nudot1'
    F1 = np.zeros(len(time_fine)) + nudot1
    
    # Take the time and create a loop:
    ti_mod = np.mod(time_fine + phi0*T, T)
    F1[(tAB < ti_mod) & (ti_mod < tAB+tBC)] = nudot2
    F1[tAB + tBC + tCD < ti_mod] = nudot2
    F1 = F1 + (nuddot * time_fine)

    # These are the constraints:
    if tAB < tCD or np.abs(nudot1) > np.abs(nudot2):
        F1 = np.zeros(len(F1)) 

    # Integrate to phase:
    F0 = integrate.cumtrapz(y=F1, x=time_fine, initial=0)
    P0 = 2 * np.pi * integrate.cumtrapz(y=F0, x=time_fine, initial=0)

    # Get the average spin-down from the phase:
    dt = time_fine[1] - time_fine[0]
    DeltaT_idx = int(DeltaT / dt)
    
    # Make it even:
    DeltaT_idx += DeltaT_idx % 2  

    tref = time_fine[0]
    time_fineprime = time_fine - tref

    time = time.reshape(len(time), 1)
    deltas = np.abs(time - time_fine)
    idxs = np.argmin(deltas, axis=1)

    vert_idx_list = idxs
    hori_idx_list = np.arange(-DeltaT_idx/2, DeltaT_idx/2, DeltaT_idx/frac_rec, 
                              dtype=int)
    A, B = np.meshgrid(hori_idx_list, vert_idx_list)
    idx_array = A + B

    time_fineprime_array = time_fineprime[idx_array]

    P0_array = P0[idx_array]

    F1_ave = np.polynomial.polynomial.polyfit(time_fineprime_array[0], 
                                              P0_array.T, 2)[2, :]/np.pi

    return F1_ave

# Assign values for the appropriate terms:
nudot1 = -3.6489 * 10**(-13)
nudot2 = -3.6635 * 10**(-13)
nuddot = 8.75 * 10**(-25)
T = 485.52 * seconds_in_day
tAB = 157.75 * seconds_in_day
tBC = 159.71 * seconds_in_day
tCD = 15.1379 * seconds_in_day
phi0 = 0.5278
sigma = 4.0932 * 10**(-16)

# Define 'plot' in order to make it easier to plot:
plot = SignalModel(time_seconds, nudot1, nudot2, nuddot, T, tAB, tBC, tCD, 
                   phi0, sigma)

# Plot the function:
fig = plt.figure(figsize=(10,6))
plt.plot(time_seconds, plot, label='SWITCHING FUNCTION')
plt.legend()
plt.title('MAGNETOSPHERIC SWITCHING FUNCTION FOR PULSAR B1828-11')
plt.xlabel('TIME (s)')
plt.ylabel('$\dot\\nu$ (Hz/s)')
txt = ("Graph showing the magnetospheric switching function, where the "
       "pulsar's spin-down switches between two states, and then repeats.")
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', 
            fontsize=10)
plt.tight_layout()
plt.savefig('switching_constant_period_function.pdf')