#!/usr/bin/python3
# plot_n0.py
#
# David Rowe Dec 2019

# Plot n0 estimates on top of speech waveforms for a few test frames
# to see if n0 estimation is working.

import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy import signal
import codec2_model

# constants

N                 = 80      # number of time domain samples in frame
width             = 256
Fs                = 8000

# read in model file records
Wo, L, A, phase, voiced = codec2_model.read(sys.argv[1])
nb_samples = Wo.size;
amp = 20.0*np.log10(A+1E-6)
# read in n0 estimates
n0_est = np.loadtxt(sys.argv[2])
print(n0_est[:20])
print("removing linear phase component....")
phase_n0_removed = np.zeros((nb_samples, width))
for i in range(nb_samples):
    for m in range(1,L[i]+1):
        a = n0_est[i]*Wo[i]*m
        phase_n0_removed[i,m] = np.angle(np.exp(1j*a))

# TODO: some how choose random set up vectors to plot.  Want them above a certain level, and mix of V and UV
frame = range(20,32)

# synthesise time domain signal
def sample_time(r):
    s = np.zeros(2*N);
    
    for m in range(1,L[r]+1):
        s = s + A[r,m]*np.cos(m*Wo[r]*range(2*N) + phase[r,m])
    return s

plot_en = 1;
if plot_en:
    plt.figure(1)
    plt.title('Amplitudes Spectra')
    for r in range(12):
        plt.subplot(3,4,r+1)
        f = frame[r];
        plt.plot(amp[f,1:L[f]],'g')
        plt.ylim(-10,60)
    plt.show(block=False)

    plt.figure(2)
    plt.title('Phase Spectra')
    for r in range(12):
        plt.subplot(3,4,r+1)
        f = frame[r];
        plt.plot(phase_n0_removed[f,1:L[f]]*180/np.pi,'r')
        plt.plot(phase[f,1:L[f]]*180/np.pi,'g')
        plt.ylim(-180,180)
    plt.show(block=False)
    
    plt.figure(3)
    plt.title('Time Domain')
    for r in range(12):
        plt.subplot(3,4,r+1)
        f = frame[r];
        s = sample_time(f)
        plt.plot(s,'g')
        mx = np.max(s)
        plt.plot([n0_est[f],n0_est[f]], [-mx/2,mx/2],'b')
    plt.show(block=False)

    # click on last figure to close all and finish
    plt.waitforbuttonpress(0)
    plt.close()
