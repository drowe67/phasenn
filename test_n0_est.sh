#!/bin/bash -x
# David Dec 2019
#

PATH=$PATH:~/codec2/build_linux/src:~/codec2/build_linux/misc

if [ "$#" -ne 1 ]; then
    n0=1
else
    n0=$1
fi
# 190 Hz is not frame syncronous
timpulse 190 $n0 filter | c2sim - --modelout imp.model
cat imp.model | est_n0 > imp_n0.txt
cat imp.model | est_n0 -r > imp_nolinear.model
./plot_n0.py imp_nolinear.model

