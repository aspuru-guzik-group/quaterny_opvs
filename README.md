# Quaterny organic photovoltaics

### High-throughput experimentation and self-driving laboratories optimize multi-component systems

This repository contains the data and experiment emulators to the demonstrated high-throughput and autonomous experimentation methods for the effective optimization of multi-component polymer blends for organic photovoltaics [1]. Specifically, the study identified photostable quarterny polymer blends composed of different ratios of PBQ-QF, P3HT, PCBM and oIDTBR (the PBQ-QF blend) or PTB7-Th, P3HT, PCBM and oIDTBR (the PTB7-Th blend).

Experimental results are reported in the file `experimental_data.xlsx`. The experiment emulators are constructed from Bayesian neural networks and can be found in the `emulators` folder of this repository.

##### Requirements

To run the experiment emulators you need
* Python 3.6
* Tensorflow 1.15
* Tensorflow Probability 0.8

### Experiencing problems?

Please create a [new issue](https://github.com/aspuru-guzik-group/quaterny_opvs/issues/new/choose) and describe your problem in detail so we can fix it.

### Reference

[1] Langner, S., Häse, F., Perea, J.D., Stubhan, T., Hauch, J., Roch, L.M., Heumueller, T., Aspuru-Guzik, A. and Brabec, C.J., 2019. Beyond Ternary OPV: High-Throughput Experimentation and Self-Driving Laboratories Optimize Multi-Component Systems. [arXiv preprint arXiv:1909.03511](https://arxiv.org/pdf/1909.03511.pdf).
