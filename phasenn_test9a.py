#!/usr/bin/python3
# phasenn_test9a.py
#
# David Rowe Nov 2019

# Estimate an impulse position from the phase spectra of a 2nd order system excited by an impulse
#
# periodic impulse train Wo at time offset n0 ->
# 2nd order system ->
# discrete phase specta ->
# NN -> single n0 output ->
# lamba layer to generate phase spectra ->
# spare loss function to compare at discrete points


import numpy as np
import sys
from keras.layers import Dense, Lambda
from keras import models,layers
from keras import initializers
import matplotlib.pyplot as plt
from scipy import signal
from keras import backend as K
import os

# be quiet tensorflow ....
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# constants

Fs                = 8000    # sample rate
N                 = 80      # number of time domain samples in frame
nb_samples        = 10000
nb_batch          = 32
nb_epochs         = 10
width             = 256
pairs             = 2*width
fo_min            = 50
fo_max            = 400
P_max             = Fs/fo_min
gain              = 2

# Generate training data

amp = np.zeros((nb_samples, width))
# phase as an angle
phase = np.zeros((nb_samples, width))
# phase encoded as cos,sin pairs:
phase_rect = np.zeros((nb_samples, pairs))
Wo = np.zeros(nb_samples)
L = np.zeros(nb_samples, dtype=int)
n0 = np.zeros(nb_samples, dtype=int)
e_rect = np.zeros((nb_samples, pairs))
target = np.zeros(nb_samples)

for i in range(nb_samples):

    # distribute fo randomly on a log scale, gives us more training
    # data with low freq frames which have more harmonics and are
    # harder to match
    r = np.random.rand(1)
    log_fo = np.log10(fo_min) + (np.log10(fo_max)-np.log10(fo_min))*r[0]
    fo = 10 ** log_fo
    #fo = fo_max
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
    n0[i] = 2
    e = np.exp(-1j*n0[i]*range(width)*np.pi/width)
    
    for m in range(1,L[i]):
        bin = int(np.round(m*Wo[i]*width/np.pi))
        mWo = bin*np.pi/width
        
        amp[i,bin] = np.log10(abs(h[m-1]))
        phase[i,bin] = np.angle(h[m-1]*e[bin])
        phase_rect[i,2*bin]   = np.cos(phase[i,bin])
        phase_rect[i,2*bin+1] = np.sin(phase[i,bin])

        # target is freq domain version of n0 in rec coords, not cos() and sin() terms
        # are in first and second half, rather than paired, to maintain compatability
        # with the custom layer
        e_rect[i,bin]       = e[bin].real
        e_rect[i,width+bin] = e[bin].imag
        target[i] = n0[i]/P_max
        
print("training data created")

# custom layer to compute a vector of DFT samples of an impulse, from
# n0.  We know how to do this with standard signal processing so we
# don't need to train layer.  However it is difficult to write signal processing
# code in "Keras backend" language
def n0_dft(n0_scaled):
    n0_scaled = K.print_tensor(n0_scaled, "n0_scaled is: ")
    n0 = n0_scaled*gain #*P_max
    n0 = K.print_tensor(n0, "n0 is: ")
    #note n0_scaled = n0/P_max such that n0_scaled stays betwen [0..1]
    N=width
    cos_term = K.cos( n0*K.cast(K.arange(N), dtype='float32')*np.pi/N)
    sin_term = K.sin(-n0*K.cast(K.arange(N), dtype='float32')*np.pi/N)
    return K.concatenate([cos_term,sin_term], axis=-1)

# testing custom layer against numpy implementation

a = layers.Input(shape=(None,))
custom_layer = K.Function([a], [n0_dft(a)])
for i in range(10):
  e_test = np.array(custom_layer([[[n0[i]/gain]]]))
  # so e_test is continuous, we just want to sample at nonzero harmonic points
  ind = np.nonzero(e_rect[i,:])
  err = (e_rect[i,ind] - e_test[0,0,ind])
  # there will be a small error as the GPU and Host don't always agree
  print(i,L[i],n0[i],err.shape, np.std(err))
  assert(np.mean(np.std(err)) < 1E-4)
print("n0_dft custom layer tested")

# custom loss function
def sparse_loss(y_true, y_pred):
    mask = K.cast( K.not_equal(y_pred, 0), dtype='float32')
    #mask = K.print_tensor(mask, "mask is: ")
    n = K.sum(mask)
    return K.sum(K.square((y_pred - y_true)*mask))/n

# testing custom loss function
x = layers.Input(shape=(None,))
y = layers.Input(shape=(None,))
loss_func = K.Function([x, y], [sparse_loss(x, y)])
#assert loss_func([[[0,1,0]], [[2,2,2]]]) == np.array([1])
#assert loss_func([[[1,1,0]], [[3,3,2]]]) == np.array([4])
print("sparse loss function tested")

# the actual NN
model = models.Sequential()
model.add(layers.Dense(pairs, activation='relu', input_dim=pairs))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1, use_bias=False))
model.add(Lambda(n0_dft))
model.summary()

from keras import optimizers
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
sgd = optimizers.SGD(lr=0.001)
model.compile(loss="mse", optimizer=sgd)
history = model.fit(e_rect, target, batch_size=nb_batch, epochs=nb_epochs)
#print(model.layers[2].get_weights()[0])
ind = np.nonzero(e_rect[0,:])
target_est = model.predict(e_rect)
print(target[:10])
print(target_est[:10])
#print(L[0],e_rect.shape, target_est.shape)
#print(e_rect[0,ind])
#print(target_est[0,ind])
quit()

# measure error in rectangular coordinates over all samples

target_est = model.predict(phase_rect)
#print(target_est)
#print(e_rect)
err = e_rect - target_est
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
    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.show(block=False)
 
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
        n0_ = target_est[r]*P_max
        print("F0: %5.1f P: %3d L: %3d n0: %3d n0_est: %5.1f" % (Wo[r]*(Fs/2)/np.pi, P, L[r], n0[r], n0_))
        plt.plot(s,'g')
        plt.plot([n0[r],n0[r]], [-25,25],'r')
        plt.plot([n0_,n0_], [-25,25],'b')
        plt.ylim(-50,50)
    plt.show(block=False)

    # click on last figure to close all and finish
    plt.waitforbuttonpress(0)
    plt.close()
