#!/usr/bin/python3
# phasenn_out.py
#
# David Rowe Dec 2019
#
# Generate phasenn output sample from an input Codec 2 model, and phaseNN .h5.
#

import numpy as np
import sys
import codec2_model
import argparse
import os
from keras.models import load_model
from keras.layers import Input, Dense, Concatenate
from keras import models,layers
from keras import initializers
from keras import backend as K

# less verbose tensorflow ....
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# constants TODO: put these is a python module 
width             = 256
pairs             = 2*width
Fs                = 8000

def list_str(values):
    return values.split(',')

parser = argparse.ArgumentParser(description='Train a NN to model Codec 2 phases')
parser.add_argument('modelin', help='Codec 2 model file in (linear phase removed)')
parser.add_argument('phasenn', help='PhaseNN trained .h5 file')
parser.add_argument('modelout', help='Codec 2 model file out (linear phase removed)')
parser.add_argument('--start', type=int, default=0, help='start frame')
parser.add_argument('--length', type=int, help='Number of frames')
args = parser.parse_args()

# read in model file records
Wo, L, A, phase, voiced = codec2_model.read(args.modelin)
nb_samples = Wo.size;
print("nb_samples: %d" % (nb_samples))

amp = np.zeros((nb_samples, width))
phase_rect = np.zeros((nb_samples, pairs))
for i in range(nb_samples):
    for m in range(1,L[i]+1):
        bin = int(np.round(m*Wo[i]*width/np.pi)); bin = min(width-1, bin)
        amp[i,bin] = np.log10(A[i,m])

# our model TODO: make a python module
model = models.Sequential()
model.add(layers.Dense(pairs, activation='relu', input_dim=width))
model.add(layers.Dense(4*pairs, activation='relu'))
model.add(layers.Dense(pairs))
model.summary()
model.load_weights(args.phasenn)

# compute rate L output phases
phase_rect_est = model.predict(amp)
phase_est = np.zeros((nb_samples, width))
st = args.start
if args.length:
    en = args.start + args.length
else:
    en = nb_samples
v = 0; uv = 0
for i in range(st,en):
    for m in range(1,L[i]+1):
        bin = int(np.round(m*Wo[i]*width/np.pi)); bin = min(width-1, bin)
        if voiced[i]:
            phase_est[i,m] = np.angle(phase_rect_est[i,2*bin] + 1j*phase_rect_est[i,2*bin+1])
            v += 1
        else:
            r = np.random.rand(1)
            phase_est[i,m] = -np.pi + 2*r[0]*np.pi
            uv += 1
print(v,uv)        
# save to output model file for synthesis
if args.modelout:
    codec2_model.write(Wo[st:en], L[st:en], A[st:en], phase_est[st:en], voiced[st:en], args.modelout)
