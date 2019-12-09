#!/usr/bin/python3
# phasenn_train.py
#
# David Rowe Dec 2019
#
# Train a NN to model phase from Codec 2 (sinusoidal model) amplitudes.
#

import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy import signal
import codec2_model
import argparse
import os
from keras.layers import Input, Dense, Concatenate
from keras import models,layers
from keras import initializers
from keras import backend as K

# less verbose tensorflow ....
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# constants

N                 = 80      # number of time domain samples in frame
width             = 256
pairs             = 2*width
Fs                = 8000
nb_batch          = 32
nb_plots          = 6

def list_str(values):
    return values.split(',')

parser = argparse.ArgumentParser(description='Train a NN to model Codec 2 phases')
parser.add_argument('modelfile', help='Codec 2 model file with linear phase removed')
parser.add_argument('--frames', type=list_str, default="30,31,32,33,34,35", help='Frames to view')
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--nnout', type=str, default="phasenn.h5", help='Name of output Codec 2 model file')
args = parser.parse_args()

assert nb_plots == len(args.frames)

# read in model file records
Wo, L, A, phase, voiced = codec2_model.read(args.modelfile)
nb_samples = Wo.size;
print("nb_samples: %d" % (nb_samples))

# set up sparse vectors, phase represented by cos(), sin() pairs
amp = np.zeros((nb_samples, width))
phase_rect = np.zeros((nb_samples, pairs))
for i in range(nb_samples):
    for m in range(1,L[i]+1):
        bin = int(np.round(m*Wo[i]*width/np.pi)); bin = min(width-1, bin)
        amp[i,bin] = np.log10(A[i,m])
        #phase_rect[i,2*bin]   = np.max((1,amp[i,bin]))*np.cos(phase[i,m])
        #phase_rect[i,2*bin+1] = np.max((1,amp[i,bin]))*np.sin(phase[i,m])
        #phase_rect[i,2*bin]   = amp[i,bin]*np.cos(phase[i,m])
        #phase_rect[i,2*bin+1] = amp[i,bin]*np.sin(phase[i,m])
        phase_rect[i,2*bin]   = np.cos(phase[i,m])
        phase_rect[i,2*bin+1] = np.sin(phase[i,m])
    
# our model
model = models.Sequential()
model.add(layers.Dense(pairs, activation='relu', input_dim=width))
model.add(layers.Dense(4*pairs, activation='relu'))
model.add(layers.Dense(pairs))
model.summary()

# custom loss function - we only care about (cos,sin) outputs at the
# non-zero positions in the sparse y_true vector.  To avoid driving the
# other samples to 0 we use a sparse loss function.  The normalisation
# term accounts for the time varying number of no-zero samples.
def sparse_loss(y_true, y_pred):
    mask = K.cast( K.not_equal(y_true, 0), dtype='float32')
    n = K.sum(mask)
    return K.sum(K.square((y_pred - y_true)*mask))/n

# testing custom loss function
y_true = Input(shape=(None,))
y_pred = Input(shape=(None,))
loss_func = K.Function([y_true, y_pred], [sparse_loss(y_true, y_pred)])
assert loss_func([[[0,1,0]], [[2,2,2]]]) == np.array([1])
assert loss_func([[[1,1,0]], [[3,2,2]]]) == np.array([2.5])
assert loss_func([[[0,1,0]], [[0,2,0]]]) == np.array([1])

# fit the model
from keras import optimizers
sgd = optimizers.SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss=sparse_loss, optimizer=sgd)
history = model.fit(amp, phase_rect, batch_size=nb_batch, epochs=args.epochs, validation_split=0.1)
model.save(args.nnout)

# measure error in angle over all samples

phase_rect_est = model.predict(amp)
phase_est = np.zeros((nb_samples, width))
used_bins = np.zeros((nb_samples, width), dtype=int)
for i in range(nb_samples):
    for m in range(1,L[i]+1):
        bin = int(np.round(m*Wo[i]*width/np.pi)); bin = min(width-1, bin)
        phase_est[i,m] = np.angle(phase_rect_est[i,2*bin] + 1j*phase_rect_est[i,2*bin+1])
        used_bins[i,m] = 1
        
ind = np.nonzero(used_bins)
c1 = np.exp(1j*phase[ind]); c2 = np.exp(1j*phase_est[ind]);
err_angle = np.angle(c1 * np.conj(c2))       
var = np.var(err_angle)
std = np.std(err_angle)
print("angle var: %4.2f std: %4.2f rads" % (var,std))
print("angle var: %4.2f std: %4.2f degs" % ((std*180/np.pi)**2,std*180/np.pi))

# synthesise time domain signal
def sample_time(r, phase):
    s = np.zeros(2*N);
    
    for m in range(1,L[r]+1):
        s = s + A[r,m]*np.cos(m*Wo[r]*range(-N,N) + phase[r,m])
    return s

frames = np.array(args.frames,dtype=int)
nb_plots = frames.size
nb_plotsy = np.floor(np.sqrt(nb_plots)); nb_plotsx=nb_plots/nb_plotsy;

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train', 'valid'], loc='upper right')
plt.title('model loss')
plt.xlabel('epoch')
plt.show(block=False)

plt.figure(2)
plt.title('Amplitudes Spectra')
for r in range(nb_plots):
    plt.subplot(nb_plotsy,nb_plotsx,r+1)
    f = frames[r];
    plt.plot(np.log10(A[f,1:L[f]]),'g')
    t = "frame %d" % (f)
    plt.title(t)
plt.show(block=False)

plt.figure(3)
plt.title('Phase Spectra')
for r in range(nb_plots):
    plt.subplot(nb_plotsy,nb_plotsx,r+1)
    f = frames[r]
    plt.plot(phase[f,1:L[f]]*180/np.pi,'g')        
    plt.plot(phase_est[f,1:L[f]]*180/np.pi,'r')        
    plt.ylim(-180,180)
    #plt.legend(("phase","phase_est"))
plt.show(block=False)
    
plt.figure(4)
plt.title('Time Domain')
for r in range(nb_plots):
    plt.subplot(nb_plotsy,nb_plotsx,r+1)
    f = frames[r];
    s = sample_time(f, phase)
    s_est = sample_time(f, phase_est)
    plt.plot(range(-N,N),s,'g')
    plt.plot(range(-N,N),s_est,'r') 
    #plt.legend(("s","s_est"))
plt.show(block=False)

print("Click on last figure to finish....")
plt.waitforbuttonpress(0)
plt.close()
