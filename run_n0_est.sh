#!/bin/bash -x
# David Dec 2019
#

PATH=$PATH:~/codec2/build_linux/src:~/codec2/build_linux/misc

if [ "$#" -ne 1 ]; then
    echo "usage: ./run_n0_est.sh rawFile"
fi
speech=$1
x=$(basename $speech)
base="${x%.*}"

c2sim $speech --modelout - | est_n0 -r > $base'_nolinear.model'
./plot_n0.py $base'_nolinear.model'
