#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 10:19:30 2022

@author: glj

This is the main script for SNR calculations for a dual energy x-ray setup.
The model consists of two polychromatic x-ray spectra incident on a 
two-material object, which are detected by either an energy-integrating
or photon-counting detector.

Signal-to-noise ratio (SNR) is calculated for each material of the object 
using estimation theory in the context of basis material decomposition. Each
signal is defined as the material's true mass thickness (density * thickness),
and each noise is defined as the square root of the Cramer-Rao Lower Bound
(CRLB) on variance. The CRLB is calculated from each detector noise model.

In this file, we consider five input spectra impinging on an object of ICRU 
tissue (40 cm) and bone (1 to 10 cm). The dose allocation between each spectral 
pair is varied from 1 to 99% of 1 uGy. SNR is calculated for each material, 
bone thickness, and dose allocation. These different conditions could be 
changed for the scenario of interest.

"""

import sys
sys.path.append('xtomosim')  # for xtomosim
from xtomosim.system import xRaySpectrum

import os
import numpy as np
from imaging_system import Material, Object, Detector, get_CRLBs_EID, get_CRLBs_PCD

# for test plots
import matplotlib.pyplot as plt


###########################################################################
### INPUTS

# run params
dose_target = 1e-6        # [Gy]
detector_mode = 'PCD'     # PCD/EID
detector_eta = None      # None or float between 0 and 1 (% photons stopped)
detector_filename = './input/detector/eta_pcd_Si_30mm.bin'   # float32 array of E, eta(E)
#E_thresh_vec = np.arange(10, 1001, 10)  # spectral detector energy thresholds [keV]
E_thresh_vec = np.arange(1010, 6000, 10)  # spectral detector energy thresholds [keV]


# label the run according to the params
run_id = f'{detector_mode}_{int(1e6*dose_target):04}uGy'
if detector_eta is not None:
    run_id = run_id + f'_{int(100*detector_eta):03}eta'

# materials (p : density [g/cm3] and t : thickness [cm])
material_1 = 'soft_tissue'
p_1 = 1.0
t_1 = 40.0
material_2 = 'bone'
p_2 = 1.85
t_2 = 1.0 
t_bones = np.arange(1,11,1.0)  # range of t_2 

# spectra, ordered in descending effective energy
spec_dir = './input/spectrum/'
spec_names = ['6MV', 'detunedMV', '140kV', '120kV', '80kV']  # all available spectra
spec_name1 = '6MV'
spec_name2 = '80kV'
dose_spec = 1e-3  # each spec file has a dose scaled to 1 mGy (center of 40-cm water cylinder)

# dose allocation range
dr = 0.01
r_vec = np.arange(dr, 1.0, dr)  



### END OF INPUTS
###########################################################################

# for saving outputs
outd = f'output/{run_id}/'

# init spectra with scaled dose to water cylinder 
spec_a = xRaySpectrum(f'{spec_dir}/{spec_name1}_1mGy_float32.bin', spec_name1)
spec_b = xRaySpectrum(f'{spec_dir}/{spec_name2}_1mGy_float32.bin', spec_name2)
spec_a.rescale_counts(dose_target / dose_spec)  
spec_b.rescale_counts(dose_target / dose_spec)  # could also use `imaging_system.get_water_dose()`


# init detector 
detector = Detector(detector_filename, detector_mode, eta=detector_eta)


if __name__ == '__main__':
    
    # create the output subdirectory    
    print(run_id)  # print check
    os.makedirs(outd, exist_ok=True)
    
    # 1 : MV-kV SNR vs. dose allocation using a vanilla PCD (no energy threshold)
    if True:
        outd_j = outd + f'{spec_a.name}_{spec_b.name}/'
        print(outd_j)
        
        os.makedirs(outd_j, exist_ok=True)
        
        # init the two material slabs and target object
        slab_1 = Material(material_1, p_1, t_1) # tissue
        for t_2 in t_bones:  # dif values of bone
            slab_2 = Material(material_2, p_2, t_2) # bone
            target = Object([slab_1, slab_2])
            
            # SNR as a function of dose allocation r, SNR(r)
            SNRs_mat1 = np.zeros(len(r_vec))   # initialize SNR vecs
            SNRs_mat2 = np.zeros(len(r_vec))
            for i, r in enumerate(r_vec):
                # get true mass thicknesses
                signals = np.array([mat.A for mat in target.materials])
                
                # rescale each spectrum to add up to total dose
                spec_a.rescale_counts(r)
                spec_b.rescale_counts(1-r)
                
                # get variance (Fisher info depends on detector type)
                if detector.mode == 'EID':
                    CRLBs = get_CRLBs_EID(spec_a, spec_b, target, detector)
                else:  # PCD
                    CRLBs = get_CRLBs_PCD(spec_a, spec_b, target, detector)
                                        
                # scale spectra back to original magnitudes 
                spec_a.rescale_counts(1/r)
                spec_b.rescale_counts(1/(1-r))
                
                # store each SNR
                SNR_mat1_r, SNR_mat2_r = signals / np.sqrt(CRLBs)    
                SNRs_mat1[i] = SNR_mat1_r
                SNRs_mat2[i] = SNR_mat2_r
    
            # save output
            SNRs_mat1.astype(np.float64).tofile(outd_j + f'mat1_{int(t_1):02}tiss_{int(t_2):02}bone.bin')
            SNRs_mat2.astype(np.float64).tofile(outd_j + f'mat2_{int(t_1):02}tiss_{int(t_2):02}bone.bin')
            
            plt.title(f'{spec_a.name} - {t_2:.0f} cm bone')
            plt.ylabel('SNR')
            plt.xlabel('dose to MV spectrum')
            plt.plot(r_vec, SNRs_mat1, 'r-', label='tissue')
            plt.plot(r_vec, SNRs_mat2, 'k-', label='bone')
            plt.axvline(r_vec[np.argmax(SNRs_mat1)], color='r')
            plt.axvline(r_vec[np.argmax(SNRs_mat2)], color='k')
            plt.legend()
            plt.show()

    # 2 : MV only, SNR vs E_thresh (one bone thickness?)
    if True:
        for E_thresh in E_thresh_vec:        
            spec = spec_a  # use the first spec [MV] only
            detector_low = Detector(detector_filename, detector_mode, eta=detector_eta, E_threshold_high=E_thresh)
            detector_high = Detector(detector_filename, detector_mode, eta=detector_eta, E_threshold_low=E_thresh)
        
            outd_j = outd + f'{spec_a.name}_spectral_{E_thresh:03}keV/'
            print(outd_j)
            
            os.makedirs(outd_j, exist_ok=True)
            
            # init the two material slabs and target object
            slab_1 = Material(material_1, p_1, t_1) # tissue
            for t_2 in t_bones:  # dif values of bone
                slab_2 = Material(material_2, p_2, t_2) # bone
                target = Object([slab_1, slab_2])
                
                # SNR as a function of dose allocation r, SNR(r)
                SNRs_mat1 = np.zeros(len(r_vec))   # initialize SNR vecs
                SNRs_mat2 = np.zeros(len(r_vec))
                for i, r in enumerate(r_vec):
                    # get true mass thicknesses
                    signals = np.array([mat.A for mat in target.materials])
                    
                    # rescale each spectrum to add up to total dose
                    spec_a.rescale_counts(r)
                    spec_b.rescale_counts(1-r)
                    
                    if detector.mode == 'EID':  # WARNING : EID doesn't make sense for spectral detection
                        print('Warning! Using EID mode with spectral detection!')
                        CRLBs = get_CRLBs_EID(spec, spec, target, detector_low, detector_high)
                    else:  
                        CRLBs = get_CRLBs_PCD(spec, spec, target, detector_low, detector_high)
                                            
                    # scale spectra back to original magnitudes 
                    spec_a.rescale_counts(1/r)
                    spec_b.rescale_counts(1/(1-r))
                    
                    # store each SNR
                    SNR_mat1_r, SNR_mat2_r = signals / np.sqrt(CRLBs)    
                    SNRs_mat1[i] = SNR_mat1_r
                    SNRs_mat2[i] = SNR_mat2_r
        
                # save output
                SNRs_mat1.astype(np.float64).tofile(outd_j + f'mat1_{int(t_1):02}tiss_{int(t_2):02}bone.bin')
                SNRs_mat2.astype(np.float64).tofile(outd_j + f'mat2_{int(t_1):02}tiss_{int(t_2):02}bone.bin')
                
            
    
    
# #%% test plots
# plt.plot(r_vec, SNRs_mat1, 'r-', label=f'{t_1:.0f} cm tissue')
# plt.plot(r_vec, SNRs_mat2, 'k-', label=f'{t_2:.0f} cm bone')
# plt.axvline(r_vec[np.argmax(SNRs_mat1)], color='r')
# plt.axvline(r_vec[np.argmax(SNRs_mat2)], color='k')
# plt.legend()
# plt.show()









        
        