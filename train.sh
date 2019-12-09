#!/bin/bash -x
# David Dec 2019
#

PATH=$PATH:~/codec2/build_linux/src:~/codec2/build_linux/misc

if [ "$#" -eq 0 ]; then
    echo "usage: ./train.sh rawFile [secs]"
fi
    
speech=$1
x=$(basename $speech)
base="${x%.*}"
if [ "$#" -eq 2 ]; then
    sox -t .sw -r 8000 -c 1 $speech -t .sw - trim 0 $2 | c2sim - --modelout - | est_n0 -r > $base'_nolinear.model'
else
    c2sim $speech --modelout - | est_n0 -r > $base'_nolinear.model'
fi
./phasenn_train.py $base'_nolinear.model' --frames 1572,1908,6792,9600,24536,25116  --epochs 10
