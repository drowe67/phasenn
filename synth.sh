#!/bin/bash -x
# David Dec 2019
#
# Putting everything back together to synthesise speech

PATH=$PATH:~/codec2/build_linux/src:~/codec2/build_linux/misc

if [ "$#" -lt 2 ]; then
    echo "usage: ./synth.sh rawFile nn.h5 [startSecs lengthSecs]"
fi

speech=$1
nn=$2
x=$(basename $speech)
base="${x%.*}"
out_model=out.model
seg=$(mktemp)'.sw'
echo $seg
if [ "$#" -lt 4 ]; then
    sox -t .sw -r 8000 -c 1 $speech -t .sw $seg
else
    st=$3
    len=$4
    sox -t .sw -r 8000 -c 1 $speech -t .sw - trim $st $len > $seg
fi    

c2sim $seg --modelout - | est_n0 -r > $base'_nolinear.model'
./phasenn_out.py $base'_nolinear.model' $nn $base'_out.model'
c2sim $seg --modelout - | est_n0 -a $base'_out.model' > $base'_comb.model'
c2sim $seg --modelin $base'_comb.model' --postfilter -o $base'_outnn.sw'

# orig speech - sinusoidal orig phases - sinusoidal phase0 - sinusoidal phaseNN
c2sim $seg -o $base'_out.sw'
c2sim $seg --phase0 --postfilter -o $base'_outp0.sw'
sox $seg $base'_out.sw' $base'_outp0.sw' $base'_outnn.sw' $base'_all.sw'
