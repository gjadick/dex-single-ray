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

import os
import numpy as np
from imaging_system import Material, Object, Source, Detector, get_CRLBs_EID, get_CRLBs_PCD


### RUN PARAMETERS
dose_target = 1e-6        # [Gy]
detector_mode = 'EID'     # PCD/EID
ideal_detector = False    # T/F, detective efficiency = 1 ?

# label the run according to the params
run_id = f'{detector_mode}_{int(1e6*dose_target):04}uGy'
if ideal_detector:
    run_id = run_id + '_ideal'
    
outd = f'output/{run_id}/'
os.makedirs(outd, exist_ok=True)

### MATERIAL INFO
### density p and thickness t
material_1 = 'soft_tissue'
p_1 = 1.0       # [g/cm^3]
t_1 = 40.0      # [cm]

material_2 = 'bone'
p_2 = 1.85      # [g/cm^3]
t_2 = 1.0       # [cm]

# detector (for detective efficiency calc)
detector_filename = './input/detector/eta.npy'   # float32 array of E, eta(E)


### SPECTRA INFO
### ordered in descending effective energy
spec_dir = './input/spectrum/'
spec_files, spec_names = np.array([
            ['Accuray_treatment6MV.csv', '6MV Treatment' ],
            ['Accuray_detuned.csv',      'MV Detuned'    ],
            ['spec140.mat',              '140kV'         ],
            ['spec120.mat',              '120kV'         ],
            ['spec80.mat',               '80kV'          ],
            ]).T
N_spec = len(spec_files)


### VARIABLES 
# dose allocation
dr = 0.01
r_vec = np.arange(dr, 1.0, dr)  

# bone thicknesses
t_bones = np.arange(1,11,1.0)




if __name__ == '__main__':
    
    # create the output subdirectory    
    print(run_id)  # print check
        
    # load the 5 source spectra and rescale to target dose
    specs = []
    water = Material('water', 1.0, 20.0)  # center of 40 cm water cylinder
    for j in range(N_spec):
        spec_j = Source(spec_dir+spec_files[j], spec_names[j])
        scale = dose_target / spec_j.get_water_dose(water)        
        spec_j.rescale_I0(scale)
        specs.append(spec_j)

    # initialize the detector
    detector = Detector(detector_filename, detector_mode, ideal_detector)
    
    # iterate over all spectral combinations
    # if specs array is ordered, then spec_a is always high energy
    for j, spec_a in enumerate(specs[:-1]):
        for jj, spec_b in enumerate(specs[j+1:]):

            # make subdirectory for each pair
            outd_j = outd + f'{spec_a.name}_{spec_b.name}/'
            print(outd_j)
            
            os.makedirs(outd_j, exist_ok=True)
            
            # initialize the two material slabs and target object
            slab_1 = Material(material_1, p_1, t_1) # tissue
            for t_2 in t_bones: # dif values of bone
                slab_2 = Material(material_2, p_2, t_2) # bone
                target = Object([slab_1, slab_2])
                
                # SNR as a function of dose allocation r, SNR(r)
                SNRs_mat1 = np.zeros(len(r_vec))   # initialize SNR vecs
                SNRs_mat2 = np.zeros(len(r_vec))
                for i, r in enumerate(r_vec):
                    # get true mass thicknesses
                    signals = np.array([mat.A for mat in target.materials])
                    
                    # rescale each spectrum to add up to total dose
                    spec_a.rescale_I0(r)
                    spec_b.rescale_I0(1-r)
                    
                    # get variance (Fisher info depends on detector type)
                    if detector.mode == 'EID':
                        CRLBs = get_CRLBs_EID(spec_a, spec_b, target, detector)
                    else:  # PCD
                        CRLBs = get_CRLBs_PCD(spec_a, spec_b, target, detector)
                                            
                    # scale spectra back to original magnitudes 
                    spec_a.rescale_I0(1/r)
                    spec_b.rescale_I0(1/(1-r))
                    
                    # store each SNR
                    SNR_mat1_r, SNR_mat2_r = signals / np.sqrt(CRLBs)    
                    SNRs_mat1[i] = SNR_mat1_r
                    SNRs_mat2[i] = SNR_mat2_r

                # save output
                SNRs_mat1.astype(np.float64).tofile(outd_j + f'mat1_{int(t_1):02}tiss_{int(t_2):02}bone.npy')
                SNRs_mat2.astype(np.float64).tofile(outd_j + f'mat2_{int(t_1):02}tiss_{int(t_2):02}bone.npy')
                
                
                
                
                
        
        