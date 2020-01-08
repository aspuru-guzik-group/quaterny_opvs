# Experiment emulators for photostability measurements of quaterny polymer blends

This folder contains two probabilistic models which emulate the photostability of two different quaterny polymer blend systems (the PBQ-QF system and the PTB7-TH system) based on high-throughput photostability measurements.

The files `emulator_pbq_qf.py` and `emulator_ptb7_th.py` implement the emulator classes and are automatically loaded with the trained set of parameters and hyperparameters. In addition, the two files contain two methods (`train` and `predict_test_set`) which demonstrate the usage of the emulator classes. 
