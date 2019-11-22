#!/usr/bin/python3
# phasenn_test8.py
#
# David Rowe Oct 2019

# Combine test8 and and test9:
#   + excite a 2nd order system with a impulse train
#   + pitch (Wo), pulse onset time (n0), 2nd order system parameters
#     (alpha and gamma) random
#   + Estimate phase spectra using the amplitude spectra and (previous) frames
#     phase spectra to extract n0.
#   + Note in this test the input phase spectra is actually the correct
#     output - but we constrict the information flowing through this part of the
#     network to ensure just n0 passes through.  Future work: it should also
#     work with other input phase spectra with the same n0

import numpy as np
import sys
from keras.layers import Input, Dense, Concatenate
from keras import Model
from keras import initializers
import matplotlib.pyplot as plt
from scipy import signal
from keras import backend as K

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
nb_samples        = 400000
nb_batch          = 32
nb_epochs         = 10
width             = 256
pairs             = 2*width
fo_min            = 50
fo_max            = 400
Fs                = 8000

# Generate training data.

amp = np.zeros((nb_samples, width))
# phase as an angle
phase = np.zeros((nb_samples, width))
# phase encoded as cos,sin pairs:
phase_rect = np.zeros((nb_samples, pairs))
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
    w,h = signal.freqz(1, [1, -2*gamma*np.cos(alpha), gamma*gamma], range(1,L[i])*Wo[i])

    # select n0 between 0...P-1 (it's periodic)
    n0[i] = r[2]*10
    e = np.exp(-1j*n0[i]*range(1,L[i])*Wo[i])

    for m in range(1,L[i]):
        bin = int(np.round(m*Wo[i]*width/np.pi))
        mWo = bin*np.pi/width
        
        amp[i,bin] = np.log10(abs(h[m-1]))
        phase[i,bin] = np.angle(h[m-1]*e[m-1])
        phase_rect[i,2*bin]   = np.cos(phase[i,bin])
        phase_rect[i,2*bin+1] = np.sin(phase[i,bin])

# estimate n0 from input phases in this part of network
phase_input = Input(shape=(pairs,), name='phase_input')
y = Dense(pairs, activation='relu')(phase_input)
y = Dense(128, activation='relu')(y)
y = Dense(1)(y)

# estimate dispersive part of phase from amplitudes in this part of network
amp_input = Input(shape=(width,), name='amp_input')
x = Dense(pairs, activation='relu')(amp_input)
x = Dense(4*pairs, activation='relu')(x)
x = Dense(pairs)(x)

# combine in final stage, should be some sort of freq dep rotation, function of n0
z = Concatenate()([y,x])
output = Dense(pairs, name='main_output')(z)
                             
model = Model(inputs=[phase_input, amp_input], outputs=[output])
model.summary()

from keras import optimizers
sgd = optimizers.SGD(lr=0.08, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss=sparse_loss, optimizer=sgd)
history = model.fit([phase_rect, amp], phase_rect, batch_size=nb_batch, epochs=nb_epochs)

# measure error in rectangular coordinates over all samples

phase_rect_est = model.predict([phase_rect, amp])
ind = np.nonzero(phase_rect)
err = (phase_rect[ind] - phase_rect_est[ind])
var = np.var(err)
std = np.std(err)
print("rect var: %f std: %f" % (var,std))

c1 = phase_rect[ind]; c1 = c1[::2] + 1j*c1[1::2]
c2 = phase_rect_est[ind]; c2 = c2[::2] + 1j*c2[1::2]
err_angle = np.angle(c1 * np.conj(c2))

var = np.var(err_angle)
std = np.std(err_angle)
print("angle var: %4.2f std: %4.2f rads" % (var,std))
print("angle var: %4.2f std: %4.2f degs" % (var*180/np.pi,std*180/np.pi))

def sample_model(r):
    phase_L = np.zeros(width, dtype=complex)
    phase_L_est = np.zeros(width, dtype=complex)
    phase_L_err = np.zeros(width, dtype=complex)
    amp_L = np.zeros(width)
    
    for m in range(1,L[r]):
        wm = m*Wo[r]
        bin = int(np.round(wm*width/np.pi))
        phase_L[m] = phase_rect[r,2*bin] + 1j*phase_rect[r,2*bin+1]
        phase_L_est[m] = phase_rect_est[r,2*bin] + 1j*phase_rect_est[r,2*bin+1]
        phase_L_err[m] = phase_L[m] * np.conj(phase_L_est[m])
        amp_L[m] = amp[r,bin]
    return phase_L, phase_L_err, amp_L
    
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
    plt.subplot(212)
    plt.hist(Wo*(Fs/2)/np.pi, bins=20)
    plt.title('phase angle error (deg) and fo (Hz)')
    plt.show(block=False)

    plt.figure(3)
    plt.title('sample vectors and error')
    for r in range(12):
        plt.subplot(3,4,r+1)
        phase, phase_err, amp_filt = sample_model(r)    
        plt.plot(np.angle(phase[1:L[r]])*180/np.pi,'g')
        plt.plot(np.angle(phase_err[1:L[r]])*180/np.pi,'r')
        plt.ylim(-180,180)
    plt.show(block=False)

    plt.figure(4)
    plt.title('filter amplitudes')
    for r in range(12):
        plt.subplot(3,4,r+1)
        phase, phase_err, amp_filt = sample_model(r)    
        plt.plot(amp_filt[1:L[r]],'g')
    plt.show(block=False)
    
    # click on last figure to close all and finish
    plt.waitforbuttonpress(0)
    plt.close()
