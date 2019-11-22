# PhaseNN

A project to model sinusoidal codec phase spectra with neural nets.

Recent breakthroughs in NN speech synthesis (WaveNet, WaveRNN, LPCNet and friends) have resulted in exciting improvements in model based synthesised speech quality.  These algorithms typically use NNs to estimate the PDF of the next speech sample using a history of previous speech samples.  This PDF is then sampled.  As such, speech is generated on a sample by sample basis.  Computational complexity is high, although steadily being reduced.

Speech codecs employing frequency domain, block based techniques such as sinusoidal transform coding can deliver high quality speech using block based synthesis.  They typically synthesise speech in blocks of 10-20ms at a time (e.g. 160-320 samples at Fs=16kHz) using efficient overlap-add IDFT techniques.  Sinusoidal codecs use a similar parameter set to NN based synthesis systems (amplitude spectra and pitch information).

However for high quality speech, sinusoidal codecs require a suitable set of the sinusoidal harmonic phases for each frame that is synthesised. This work aims to generate the sinusoid phases from amplitude information using NNs, in order to develop a block based NN synthesis engine based on sinusoidal coding.

## Status (Nov 2019)

Building up techniques for modelling phase using NNs and toy speech models (2nd order filters) in a series of tests.

