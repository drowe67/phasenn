#!/usr/bin/python3
# plot_n0.py
#
# David Rowe Dec 2019

# Plot n0 estimates on top of speech waveforms for a few test frames
# to see if n0 estimation is working.

'''
Usage:
~codec/build_linux$ ./misc/timpulse 1 | ./src/c2sim - --modelout imp.model
~codec/build_linux$ ./misc/timpulse 1 | ./src/c2sim - --modelout - | ./misc/est_n0 > imp_n0.txt
~phasenn$ ./plot_n0.py ~/codec2/build_linux/imp.model ~/codec2/build_linux/imp_n0.txt

Green line (phase_n0_removed) should be flat, as the phase of an
impluse train is comprised of just a linear time shift component, and 0
dispersive component.

'''

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

'''
# Python version of est_n0.c
n0_est2 =  np.zeros((nb_samples))
print("estimating linear phase component...")
for i in range(nb_samples):
    err_min = 1E32
    P = 2*np.pi/Wo[i]
    
    for test_n0 in np.arange(0,P,0.25):
        e = np.exp(1j*test_n0*np.arange(1,L[i]+1)*Wo[i])
        err = np.dot(np.log10(A[i,1:L[i]+1]), np.abs(np.exp(1j*phase[i,1:L[i]+1]) - e)**2)
        if err < err_min:
            err_min = err
            n0_est2[i] = test_n0
print(n0_est2[:10])
'''
print(n0_est[:10])

print("removing linear phase component....")
phase_linear = np.zeros((nb_samples, width))
phase_n0_removed = np.zeros((nb_samples, width))
for i in range(nb_samples):
    for m in range(1,L[i]+1):
        phase_linear[i,m] = n0_est[i]*Wo[i]*m
        phase_n0_removed[i,m] = phase[i,m] - phase_linear[i,m]
        phase_n0_removed[i,m] = np.angle(np.exp(1j*phase_n0_removed[i,m]))

# TODO: some how choose random set up vectors to plot.  Want them above a certain level, and mix of V and UV
frame = range(160,172)

# synthesise time domain signal
def sample_time(r):
    s = np.zeros(2*N);
    
    for m in range(1,L[r]+1):
        s = s + A[r,m]*np.cos(m*Wo[r]*range(-N,N) + phase[r,m])
    return s

nb_plots = 8; nb_plotsy = np.floor(np.sqrt(nb_plots)); nb_plotsx=nb_plots/nb_plotsy;
plot_en = 1; plot_all_phases = 0;
if plot_en:
    plt.figure(1)
    plt.title('Amplitudes Spectra')
    for r in range(nb_plots):
        plt.subplot(nb_plotsy,nb_plotsx,r+1)
        f = frame[r];
        plt.plot(amp[f,1:L[f]],'g')
        #plt.ylim(-10,60)
    plt.show(block=False)

    plt.figure(2)
    plt.title('Phase Spectra')
    for r in range(nb_plots):
        plt.subplot(nb_plotsy,nb_plotsx,r+1)
        f = frame[r]
        if plot_all_phases:
            plt.plot(phase[f,1:L[f]]*180/np.pi,'g')
            plt.plot(np.angle(np.exp(1j*phase_linear[f,1:L[f]]))*180/np.pi,'b')
        plt.plot(phase_n0_removed[f,1:L[f]]*180/np.pi,'r')
        plt.ylim(-180,180)
    plt.show(block=False)
    print("Green: input phase, Blue: est linear, Red: phase with linear removed")
    
    plt.figure(3)
    plt.title('Time Domain')
    for r in range(nb_plots):
        plt.subplot(nb_plotsy,nb_plotsx,r+1)
        f = frame[r];
        s = sample_time(f)
        plt.plot(range(-N,N),s,'g')
        mx = np.max(np.abs(s))
        #print(r,f,n0_est[f])
        # so our centre is n0_est in advance, actual n0 is n0_est behind centre.  I think.
        plt.plot([-n0_est[f],-n0_est[f]], [-mx/2,mx/2],'b')
    plt.show(block=False)

    # click on last figure to close all and finish
    plt.waitforbuttonpress(0)
    plt.close()
