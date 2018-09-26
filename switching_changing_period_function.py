# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 09:26:33 2018

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
# Add in the variables 'T_0' and 'Tdot':
def SignalModel(time, nudot1, nudot2, nuddot, T_0, Tdot, rA, rB, rC, phi0, 
                sigma):

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
   
    # This is the changing period function:
    T = (T_0 + (Tdot * (time_fine)))

    # Take the time and create a loop:
    # Fix the arrays:
    ti_mod = np.mod(time_fine + phi0*T, T) / T

    F1[(rA < ti_mod) & (ti_mod < rA+rB)] = nudot2
    F1[rA + rB + rC < ti_mod] = nudot2
    F1 = F1 + nuddot * time_fine

    # These are the constraints:
    if rA < rC or np.abs(nudot1) > np.abs(nudot2):
        F1 = np.zeros(len(F1))
    
    if (rA + rB + rC) > 1:
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
phi0 = 0.5278
sigma = 4.0932 * 10**(-16)

T_0 = 485.52 * seconds_in_day
Tdot = -0.01

rA = 0.4
rB = 0.3
rC = 0.05

# Define 'plot' in order to make it easier to plot:
plot = SignalModel(time_seconds, nudot1, nudot2, nuddot, T_0, Tdot, rA, rB, rC, 
                   phi0, sigma)

# Plot the function:
fig = plt.figure(figsize=(10,6))
plt.plot(time_seconds, plot, label='SWITCHING FUNCTION WITH CHANGING PERIOD')
plt.legend()
plt.title('MAGNETOSPHERIC SWITCHING FUNCTION, WITH CHANGING PERIOD, FOR PULSAR'
          ' B1828-11')
plt.xlabel('TIME (s)')
plt.ylabel('$\dot\\nu$ (Hz/s)')
txt = ("Graph showing the magnetospheric switching function, where the period "
       "is changing.")
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', 
            fontsize=10)
plt.tight_layout()
plt.savefig('switching_changing_period_function.pdf')