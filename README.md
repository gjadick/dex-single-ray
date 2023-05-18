# dex-single-ray

Signal detection framework for a single dual-energy x-ray through a 
two-material object. The dual signals enable the possibility for material 
decomposition, so this framework can be used for computing an idealized
basis material signal-to-noise ratio (SNR). In this script, SNR is computed
as the ratio of each material's true mass-thickness (density*thickness) to
the square root of the Cramer-Rao Lower Bound (CRLB) on variance, which
is found using estimation theory.


## main.py
Main file for running the single-line model. Parameters to set:
- dose_target: total dose delivered by both spectra in Gy
- detector_mode: energy integrating (EID) or photon counting (PCD)
- ideal_detector: whether detective efficiency is idealized = 1
- material data for the two slabs and detector (density, thickness, name)
- spectrum files and their IDs for labeling
- dose allocation array (r = 1 to 99%)
- bone thickness array (t_bones = 1 to 10 cm)


## imaging_system.py
Helper classes and functions for setting up the estimation theory framework
and computing the Cramer-Rao Lower Bound.


## plots.py
Script used to generate plots to analyze data output from main.



