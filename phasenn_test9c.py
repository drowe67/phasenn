#!/usr/bin/python3
# phasenn_test9c.py
#
# David Rowe Nov 2019

# Estimate an impulse position from the phase spectra of a 2nd order system excited by an impulse
#
# periodic impulse train Wo at time offset n0 -> 2nd order system -> discrete phase specta -> NN -> n0
#
# This version uses regular DSP rather than a NN to estimate n0

import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy import signal

# constants

Fs                = 8000
N                 = 80      # number of time domain samples in frame
nb_samples        = 1000
width             = 256
pairs             = 2*width
fo_min            = 50
fo_max            = 400
P_max             = Fs/fo_min

# Generate training data

amp = np.zeros((nb_samples, width))
# phase as an angle
phase = np.zeros((nb_samples, width))
# phase encoded as cos,sin pairs:
phase_rect = np.zeros((nb_samples, pairs))
Wo = np.zeros(nb_samples)
L = np.zeros(nb_samples, dtype=int)
n0 = np.zeros(nb_samples, dtype=int)
target = np.zeros((nb_samples,1))
e_rect = np.zeros((nb_samples, pairs))

for i in range(nb_samples):

    # distribute fo randomly on a log scale, gives us more training
    # data with low freq frames which have more harmonics and are
    # harder to match
    r = np.random.rand(1)
    log_fo = np.log10(fo_min) + (np.log10(fo_max)-np.log10(fo_min))*r[0]
    fo = 10 ** log_fo
    Wo[i] = fo*2*np.pi/Fs
    L[i] = int(np.floor(np.pi/Wo[i]))
    # pitch period in samples
    P = 2*L[i]

    r = np.random.rand(3)

    # sample 2nd order IIR filter with random peak freq (alpha) and peak amplitude (gamma)
    alpha = 0.1*np.pi + 0.4*np.pi*r[0]
    gamma = 0.9 + 0.09*r[1]
    w,h = signal.freqz(1, [1, -2*gamma*np.cos(alpha), gamma*gamma], range(1,L[i])*Wo[i])
    
    # select n0 between 0...P-1 (it's periodic)
    n0[i] = r[2]*P
    #n0[i] = 10
    e = np.exp(-1j*n0[i]*range(width)*np.pi/width)
    
    for m in range(1,L[i]):
        bin = int(np.round(m*Wo[i]*width/np.pi))
        
        amp[i,bin] = np.log10(abs(h[m-1]))
        phase[i,bin] = np.angle(h[m-1]*e[bin])
        #phase[i,bin] = np.angle(e[bin])
        phase_rect[i,2*bin]   = np.cos(phase[i,bin])
        phase_rect[i,2*bin+1] = np.sin(phase[i,bin])

        # target is n0 in rec coords                      
        target[i] = n0[i]
        
# use regular DSP to estimate n0

target_est =  np.zeros((nb_samples,1))
for i in range(nb_samples):
    err_min = 1E32
    P = 2*L[i]
    for test_n0 in np.arange(0,P,0.25):
        e = np.exp(-1j*test_n0*np.arange(width)*np.pi/width)
        err = 0.0
        for m in range(1,L[i]):
            bin = int(np.round(m*Wo[i]*width/np.pi))
            err = err + (10**amp[i,bin])*(np.abs(np.exp(1j*phase[i,bin]) - e[bin])**2)
        if err < err_min:
            err_min = err
            target_est[i] = test_n0
        #print(i,test_n0, err, err_min)
        
# measure error in rectangular coordinates over all samples

err = target - target_est
var = np.var(err)
std = np.std(err)
print("var: %f std: %f" % (var,std))

def sample_freq(r):
    phase_L = np.zeros(L[r], dtype=complex)
    amp_L = np.zeros(L[r])
    
    for m in range(1,L[r]):
        wm = m*Wo[r]
        bin = int(np.round(wm*width/np.pi))
        phase_L[m] = phase_rect[r,2*bin] + 1j*phase_rect[r,2*bin+1]
        amp_L[m] = amp[r,bin]
    return phase_L, amp_L

# synthesise time domain signal
def sample_time(r):
    s = np.zeros(2*N);
    
    for m in range(1,L[r]):
        wm = m*Wo[r]
        bin = int(np.round(wm*width/np.pi))
        Am = 10 ** amp[r,bin]
        phi_m = np.angle(phase_rect[r,2*bin] + 1j*phase_rect[r,2*bin+1])
        s = s + Am*np.cos(wm*(range(2*N)) + phi_m)
    return s

plot_en = 1;
if plot_en:
    plt.figure(2)
    plt.hist(err, bins=20)
    plt.show(block=False)

    plt.figure(3)
    plt.plot(target[:12],'b')
    plt.plot(target_est[:12],'g')
    plt.show(block=False)

    plt.figure(4)
    plt.title('Freq Domain')
    for r in range(12):
        plt.subplot(3,4,r+1)
        phase_L, amp_L = sample_freq(r)
        plt.plot(20*amp_L,'g')
        plt.ylim(-20,20)
    plt.show(block=False)

    plt.figure(5)
    plt.title('Time Domain')
    for r in range(12):
        plt.subplot(3,4,r+1)
        s = sample_time(r)
        n0_ = target_est[r]
        print("F0: %5.1f P: %3d L: %3d n0: %3d n0_est: %5.1f" % (Wo[r]*(Fs/2)/np.pi, P, L[r], n0[r], n0_))
        plt.plot(s,'g')
        plt.plot([n0[r],n0[r]], [-25,25],'r')
        plt.plot([n0_,n0_], [-25,25],'b')
        plt.ylim(-50,50)
    plt.show(block=False)

    # click on last figure to close all and finish
    plt.waitforbuttonpress(0)
    plt.close()
