#!/bin/bash -x
# David Dec 2019
#
# Putting everything back together to synthesise speech

PATH=$PATH:~/codec2/build_linux/src:~/codec2/build_linux/misc

if [ "$#" -ne 4 ]; then
    echo "usage: ./synth.sh rawFile nn.h5 startSecs lengthSecs"
fi
speech=$1
nn=$2
st=$3
len=$4
x=$(basename $speech)
base="${x%.*}"
out_model=out.model

sox -t .sw -r 8000 -c 1 $speech -t .sw - trim $st $len | c2sim - --modelout - | est_n0 -r > $base'_nolinear.model'
./phasenn_out.py $base'_nolinear.model' $nn $base'_out.model'
sox -t .sw -r 8000 -c 1 $speech -t .sw - trim $st $len | c2sim - --modelout - | est_n0 -a $base'_out.model' > $base'_comb.model'
sox -t .sw -r 8000 -c 1 $speech -t .sw - trim $st $len | c2sim - --modelin $base'_comb.model' -o $base'_outnn.sw'
sox -t .sw -r 8000 -c 1 $speech -t .sw - trim $st $len | c2sim - -o $base'_out.sw'
sox -t .sw $base'_outnn.sw' -t .sw $base'_out.sw' $base'_both.sw'
