#!/bin/bash -x
# David Dec 2019
#

PATH=$PATH:~/codec2/build_linux/src:~/codec2/build_linux/misc

speech=~/Downloads/train_8k.sw
x=$(basename $speech)
base="${x%.*}"

sox -t .sw -r 8000 -c 1 $speech -t .sw - trim 0 600 | c2sim - --modelout - | est_n0 -r > $base'_nolinear.model'
./phasenn_train.py $base'_nolinear.model' --frames 1572,1908,6792,9600
