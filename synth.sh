#!/bin/bash -x
# David Dec 2019
#
# Putting everything back together to synthesise speech

PATH=$PATH:~/codec2/build_linux/src:~/codec2/build_linux/misc

if [ "$#" -ne 1 ]; then
    echo "usage: ./synth.sh rawFile"
fi
speech=$1
x=$(basename $speech)
base="${x%.*}"
out_model=out.model

sox -t .sw -r 8000 -c 1 $speech -t .sw - trim 0 3 | c2sim - --modelout - | est_n0 -a $out_model > $base'_comb.model'
sox -t .sw -r 8000 -c 1 $speech -t .sw - trim 0 3 | c2sim - --modelin $base'_comb.model' -o $base'_out.raw'
