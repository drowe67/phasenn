# PhaseNN

A project to model sinusoidal codec phase spectra with neural nets.

## Introduction

Recent breakthroughs in (Neural Net) NN speech synthesis (WaveNet, WaveRNN, LPCNet and friends) have resulted in exciting improvements in model based synthesised speech quality.  These algorithms typically use NNs to estimate the PDF of the next speech sample conditioned on input features and a history of previously synthesised speech samples.  This PDF is then sampled to obtain the next output speech sample.  As the algorithms need all previous output speech samples, speech must be generated on a sample by sample basis.  Computational complexity is high, although steadily being reduced.

Speech codecs employing frequency domain, block based techniques such as sinusoidal transform coding can deliver high quality speech using block based synthesis.  They typically synthesise speech in blocks of 10-20ms at a time (e.g. 160-320 samples at Fs=16kHz) using efficient overlap-add IDFT techniques.  Sinusoidal codecs use a similar parameter set to the features used for NN based synthesis systems (some form of amplitude spectra, pitch information, voicing).

For high quality speech, sinusoidal codecs require a suitable set of the sinusoidal harmonic phases for each frame that is synthesised. This work aims to generate the sinusoid phases from amplitude information using NNs, in order to develop a block based NN synthesis engine based on sinusoidal coding.

## Status (Dec 2019)

Building up techniques for modelling phase using NNs and toy speech models (cascades of 2nd order filters excited by impulse trains) in a series of tests.

## Challenges

1. We must represent phase (angles) in a NN.  Phase is being represnted by (cos(angle), sin(angle)) pairs, which when trained, tend to develop weights that behave like complex numbers/matrix rotations.
1. The number of phases in the sinsuoidal model is time varying, based on the pitch of the current frame.  This is being addressed by mapping the frequency of each sinusoid onto the index a sparse, fixed length vector. 

## Example

Here is the output from [phasenn_test11.py](phasenn_test11.py).  The first plot is a series of magnitude spectra of simulated speech frames.  The voiced frames have two fairly sharp peaks (formants) beneath Fs/2 with structured phase consisting of linear and dispersive terms.  Unvoiced frames have less sharp peaks above Fs/2, and random phases.

![](example_mag.png "Magnitude Spectra")
![](example_phase.png "Phase Spectra")

The next plot shows the disperive component of the original phase spectra (green), the hase spectra with an estimate of the linear phase term removed (red), and the NN output estimated dispersive phase (blue).  For voiced frames, we would like green (original) and blue (NN estimate) to match.  In particular we want to model the rapid phase shift across the peak of the amplitude spectra - this is the dispersive term that shifts the phase of high energy speech harmonics apart and reduces the buzzy/unnatural quality in synthsised speech.

When training from real world data, we have frames of phase spectra with the linear and disperive phase components combined. We will not know the linear term, and therefore must estimate it. This simulation introduces small errors in the linear term estimation (+/-1 sample), which can lead to large phase differences at high frequencies.  The red (original phase with estimated linear term removed) diverges from the true dispersive input (green), as the estimation of the linear term is not perfect.  However over the training database these errors tend to have a zero mean - this simulation suggests they are being "trained out" by the NN, resulting in a reasonable model of the dispersive term (blue), albiet with some estimation "noise" at high frequencies.  This HF noise may be useful, as it matches the lack of structure of HF phases in real speech.

For unvoiced speech, we want the NN output (blue) to be random.  They do not need to match the original input phase spectra. The NN appears to presever this random phase structure in this simulation.  This may remove the need for a voicing estimate - voicing can be deduced from the magnitude spectra.

