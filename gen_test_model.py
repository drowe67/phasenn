#!/usr/bin/python3
# gen_test_model.py
#
# David Rowe Dec 2019
#
# Generate some contrived pusedo-speech test data for driving phasenn_train.py 

import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy import signal
import codec2_model

# constants

N                 = 80      # number of time domain samples in frame
width             = 256
nb_samples        = 100000
fo_min            = 50
fo_max            = 400
Fs                = 8000

# Generate training data.

print("Generate training data")

# amplitude and phase at rate L
A = np.zeros((nb_samples, width))
phase = np.zeros((nb_samples, width))

# side information
Wo = np.zeros(nb_samples)
L = np.zeros(nb_samples, dtype=int)
voiced = np.zeros(nb_samples, dtype=int)

for i in range(nb_samples):

    # distribute fo randomly on a log scale, gives us more training
    # data with low freq frames which have more harmonics and are
    # harder to match
    r = np.random.rand(1)
    log_fo = np.log10(fo_min) + (np.log10(fo_max)-np.log10(fo_min))*r[0]
    fo = fo_min
    fo = 10 ** log_fo
    Wo[i] = fo*2*np.pi/Fs
    L[i] = int(np.floor(np.pi/Wo[i]))
    # pitch period in samples
    P = 2*L[i]
 
    r = np.random.rand(1)
    voiced[i] = r[0] > 0.1
    
    # sample 2nd order IIR filter with random peak freq

    r1 = np.random.rand(2)
    r2 = np.random.rand(2)
    if voiced[i]:
        # choose alpha and gamma to get something like voiced speech
        alpha1 = 0.05*np.pi + 0.25*np.pi*r1[0]
        gamma1 = 0.9 + 0.09*r1[1]
        alpha2 = alpha1 + 0.4*np.pi*r2[0]
        gamma2 = 0.9 + 0.05*r2[1]
        gain = 10
    else:
        alpha1 = 0.5*np.pi + 0.4*np.pi*r1[0]
        gamma1 = 0.8 + 0.1*r1[1]
        alpha2 = 0.5*np.pi + 0.4*np.pi*r2[0]
        gamma2 = 0.8 + 0.1*r2[1]
        gain = 1
        
    w1,h1 = signal.freqz(gain, [1, -2*gamma1*np.cos(alpha1), gamma1*gamma1], range(1,L[i]+1)*Wo[i])
    w2,h2 = signal.freqz(gain, [1, -2*gamma2*np.cos(alpha2), gamma2*gamma2], range(1,L[i]+1)*Wo[i])
    
    for m in range(1,L[i]+1):
        A[i,m] = np.abs(h1[m-1]*h2[m-1])
        if voiced[i]:
            phase[i,m] = np.angle(h1[m-1]*h2[m-1])
        else:
            r = np.random.rand(1)    
            phase[i,m] = -np.pi + r[0]*2*np.pi
                        
# write to a Codec 2 model file

codec2_model.write(Wo, L, A, phase, voiced, "gen_test.model")
