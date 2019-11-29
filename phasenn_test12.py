#!/usr/bin/python3
# phasenn_test12.py
#
# David Rowe Nov 2019

# Try to use a NN to extract linear phase (n0 ) term, leaving just dispersive

# Combine test8 and and test9c:
#   + excite a 2nd order system with a impulse train
#   + pitch (Wo), pulse onset time (n0), 2nd order system parameters
#     (alpha and gamma) random
#   + see if we can train to resolve just dispersive phase term

import numpy as np
import sys
from keras.layers import Input, Dense, Concatenate
from keras import models,layers
from keras import initializers
import matplotlib.pyplot as plt
from scipy import signal
from keras import backend as K
# less verbose tensorflow ....
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# custom loss function
def sparse_loss(y_true, y_pred):
    mask = K.cast( K.not_equal(y_pred, 0), dtype='float32')
    n = K.sum(mask)
    return K.sum(K.square((y_pred - y_true)*mask))/n

# testing custom loss function
x = Input(shape=(None,))
y = Input(shape=(None,))
loss_func = K.Function([x, y], [sparse_loss(x, y)])
assert loss_func([[[1,1,1]], [[0,2,0]]]) == np.array([1])
assert loss_func([[[0,1,0]], [[0,2,0]]]) == np.array([1])

# constants

N                 = 80      # number of time domain samples in frame
nb_samples        = 100000
nb_batch          = 32
nb_epochs         = 25
width             = 256
pairs             = 2*width
fo_min            = 50
fo_max            = 400
Fs                = 8000

# Generate training data.

print("Generate training data")

# amplitude and phase at rate L
amp = np.zeros((nb_samples, width))
phase_disp = np.zeros((nb_samples, width))
phase_comb = np.zeros((nb_samples, width))

# rate "width" sparse phase vectors encoded as cos,sin pairs:
phase_disp_rect = np.zeros((nb_samples, pairs))
phase_comb_rect = np.zeros((nb_samples, pairs))

# side information
Wo = np.zeros(nb_samples)
L = np.zeros(nb_samples, dtype=int)
n0 = np.zeros(nb_samples, dtype=int)

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
 
    r = np.random.rand(3)
    
    # sample 2nd order IIR filter with random peak freq, choose alpha
    # and gamma to get something like voiced speech
    alpha = 0.1*np.pi + 0.4*np.pi*r[0]
    gamma = 0.9 + 0.09*r[1]
    w,h = signal.freqz(1, [1, -2*gamma*np.cos(alpha), gamma*gamma], range(1,L[i]+1)*Wo[i])

    # select n0 between 0...P-1 (it's periodic)
    n0[i] = r[2]*P
    e = np.exp(-1j*n0[i]*range(1,L[i]+1)*Wo[i])

    for m in range(1,L[i]+1):
        amp[i,m] = np.log10(np.abs(h[m-1]))
        phase_comb[i,m] = np.angle(h[m-1]*e[m-1])
        phase_disp[i,m] = np.angle(h[m-1])

        bin = int(np.round(m*Wo[i]*width/np.pi)); bin = min(width-1, bin)
        phase_disp_rect[i,2*bin]   = np.cos(phase_disp[i,m])
        phase_disp_rect[i,2*bin+1] = np.sin(phase_disp[i,m])
        phase_comb_rect[i,2*bin]   = np.cos(phase_comb[i,m])
        phase_comb_rect[i,2*bin+1] = np.sin(phase_comb[i,m])
    
model = models.Sequential()
model.add(layers.Dense(4*pairs, activation='relu', input_dim=pairs))
model.add(layers.Dense(4*pairs, activation='relu'))
model.add(layers.Dense(pairs))
model.summary()

from keras import optimizers
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss=sparse_loss, optimizer=sgd)
history = model.fit(phase_comb_rect, phase_disp_rect, batch_size=nb_batch, epochs=nb_epochs)

# measure error in angle over all samples

phase_disp_est_rect = model.predict(phase_comb_rect)
phase_disp_est = np.zeros((nb_samples, width))
used_bins = np.zeros((nb_samples, width), dtype=int)
for i in range(nb_samples):
    for m in range(1,L[i]+1):
        bin = int(np.round(m*Wo[i]*width/np.pi)); bin = min(width-1, bin)
        phase_disp_est[i,m] = np.angle(phase_disp_est_rect[i,2*bin] + 1j*phase_disp_est_rect[i,2*bin+1])
        used_bins[i,m] = 1
        
ind = np.nonzero(used_bins)
c1 = np.exp(1j*phase_disp[ind]); c2 = np.exp(1j*phase_disp_est[ind]);
err_angle = np.angle(c1 * np.conj(c2))       
var = np.var(err_angle)
std = np.std(err_angle)
print("angle var: %4.2f std: %4.2f rads" % (var,std))
print("angle var: %4.2f std: %4.2f degs" % (var*180/np.pi,std*180/np.pi))

plot_en = 1;
if plot_en:
    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.show(block=False)
 
    plt.figure(2)
    plt.subplot(211)
    plt.hist(err_angle*180/np.pi, bins=20)
    plt.title('phase angle error (deg) and fo (Hz)')
    plt.subplot(212)
    plt.hist(Wo*(Fs/2)/np.pi, bins=20)
    plt.show(block=False)

    plt.figure(3)
    plt.title('filter amplitudes')
    for r in range(12):
        plt.subplot(3,4,r+1)
        plt.plot(amp[r,:L[r]],'g')
    plt.show(block=False)

    plt.figure(4)
    plt.title('sample vectors and error')
    for r in range(12):
        plt.subplot(3,4,r+1)
        plt.plot(phase_disp[r,:L[r]]*180/np.pi,'g')
        plt.plot(phase_disp_est[r,:L[r]]*180/np.pi,'r')
        #plt.plot(phase_est[r,:L[r]]*180/np.pi,'r')
        plt.ylim(-180,180)
    plt.show(block=False)
    
    # click on last figure to close all and finish
    plt.waitforbuttonpress(0)
    plt.close()
