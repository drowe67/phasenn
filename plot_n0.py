#!/usr/bin/python3
# plot_n0.py
#
# David Rowe Dec 2019

# Generate some plots Plot n0 estimates on top of speech waveforms for a few test frames
# to see if n0 estimation is working.

'''
Sample usage for a contrived impulse train signal:

Generate an impulse train with time offset n0 1, and extract codec 2 model records:
  ~codec/build_linux$ ./misc/timpulse 1 | ./src/c2sim - --modelout ~/phasenn/imp.model
Estimate n0:
  ~codec/build_linux$ ./misc/timpulse 1 | ./src/c2sim - --modelout - | ./misc/est_n0 > ~/phasenn/imp_n0.txt
Remove linear phase component due to time offset n0:
  ~codec/build_linux$ ./misc/timpulse 1 | ./src/c2sim - --modelout - | ./misc/est_n0 -r > ~/phasenn/imp_nolinear.model

Usage (this script removes linear):
  ~phasenn$ ./plot_n0.py imp.model -n0 imp_n0.txt
Usage (linear removed by C program):
  ~phasenn$ ./plot_n0.py imp_nolinear.model
Usage (start plots at frame 30):
  ~phasenn$ ./plot_n0.py --start 30 imp_nolinear.model

Green line (phase_n0_removed) should be flat, as the phase of an
impulse train is comprised of just a linear time shift component, and 0
dispersive component.

'''

import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy import signal
import codec2_model
import argparse

# constants

N                 = 80      # number of time domain samples in frame
width             = 256
Fs                = 8000

parser = argparse.ArgumentParser(description='Plot phase spectra and synthesised speech')
parser.add_argument('modelfile', help='Codec 2 model file')
parser.add_argument('--n0file', help='text file of n0 estimates')
parser.add_argument('--start', type=int, default=30, help=' start frame')

args = parser.parse_args()
# read in model file records
Wo, L, A, phase, voiced = codec2_model.read(args.modelfile)
nb_samples = Wo.size;
amp = 20.0*np.log10(A+1E-6)

# read in n0 estimates
have_n0 = 0
if args.n0file:
  n0_est = np.loadtxt(args.n0file)
  have_n0 = 1
  print(n0_est[:10])
  
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

if have_n0:
    print("removing linear phase component....")
    phase_linear = np.zeros((nb_samples, width))
    phase_n0_removed = np.zeros((nb_samples, width))
    for i in range(nb_samples):
        for m in range(1,L[i]+1):
            phase_linear[i,m] = n0_est[i]*Wo[i]*m
            phase_n0_removed[i,m] = phase[i,m] - phase_linear[i,m]
            phase_n0_removed[i,m] = np.angle(np.exp(1j*phase_n0_removed[i,m]))

# synthesise time domain signal
def sample_time(r):
    s = np.zeros(2*N);
    
    for m in range(1,L[r]+1):
        s = s + A[r,m]*np.cos(m*Wo[r]*range(-N,N) + phase[r,m])
    return s

nb_plots = 4; nb_plotsy = np.floor(np.sqrt(nb_plots)); nb_plotsx=nb_plots/nb_plotsy;
startframe = args.start
frame = range(startframe,startframe+nb_plots)

plt.figure(1)
plt.title('Amplitudes Spectra')
for r in range(nb_plots):
    plt.subplot(nb_plotsy,nb_plotsx,r+1)
    f = frame[r];
    plt.plot(amp[f,1:L[f]],'g')
plt.show(block=False)

plt.figure(2)
plt.title('Phase Spectra')
for r in range(nb_plots):
    plt.subplot(nb_plotsy,nb_plotsx,r+1)
    f = frame[r]
    if have_n0:
        plt.plot(phase[f,1:L[f]]*180/np.pi,'g')
        plt.plot(np.angle(np.exp(1j*phase_linear[f,1:L[f]]))*180/np.pi,'b')
        plt.plot(phase_n0_removed[f,1:L[f]]*180/np.pi,'r')
    else:
        plt.plot(phase[f,1:L[f]]*180/np.pi,'r')        
    plt.ylim(-180,180)
plt.show(block=False)
if have_n0:
    print("Green: input phase, Blue: est linear, Red: phase with linear removed")
else:
    print("Red: phase with linear removed")
    
plt.figure(3)
plt.title('Time Domain')
for r in range(nb_plots):
    plt.subplot(nb_plotsy,nb_plotsx,r+1)
    f = frame[r];
    s = sample_time(f)
    plt.plot(range(-N,N),s,'g')
    mx = np.max(np.abs(s))
    # so our centre is n0_est in advance, actual n0 is n0_est behind centre.  I think.
    if have_n0:
        plt.plot([-n0_est[f],-n0_est[f]], [-mx/2,mx/2],'b')
    else:
        plt.plot([0,0], [-mx/2,mx/2],'b')
        
plt.show(block=False)

print("Click on last figure to finish....")
plt.waitforbuttonpress(0)
plt.close()
