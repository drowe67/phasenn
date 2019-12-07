#!/bin/bash -x
# David Dec 2019
#

PATH=$PATH:~/codec2/build_linux/src:~/codec2/build_linux/misc

speech=~/Downloads/all_8k.sw
x=$(basename $speech)
base="${x%.*}"

c2sim $speech --modelout - | est_n0 -r > $base'_nolinear.model'
./phasenn_train.py $base'_nolinear.model' --frames 560,655,990,2899
