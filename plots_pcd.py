#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 15:10:10 2023

@author: giavanna
"""

import sys
sys.path.append('xtomosim')  
from xtomosim.system import xRaySpectrum
# from imaging_system import Detector

import os
import numpy as np
import matplotlib.pyplot as plt

# define / load the parameters to use 
from main import E_thresh_vec, r_vec, t_1, t_bones, dose_target, dose_spec, \
                 spec_name1, spec_name2, spec_dir, \
                 detector_filename, detector_mode, detector_eta

from plots import bf, label_panels, heatmap
 

outd_eid = f'output/EID_{int(1e6*dose_target):04}uGy/'
outd_pcd = f'output/PCD_{int(1e6*dose_target):04}uGy/'
figd = f'output/figs/detcomp_{int(1e6*dose_target):04}uGy/'
savefig = False
if savefig:
    os.makedirs(figd, exist_ok=True)
    
# for uniform figure formatting
plt.rcParams.update({
    # figure
    "figure.dpi": 600,
    # text
    "font.size":10,
    "font.family": "serif",
    "font.serif": ['Computer Modern Roman'],
    "text.usetex": True,
    # axes
    "axes.titlesize":10,
    "axes.labelsize":8,
    "axes.linewidth": 1,
    # ticks
    "xtick.top": True,
    "ytick.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.labelsize":8,
    "ytick.labelsize":8,
    # grid
    "axes.grid" : False,
    "axes.grid.which" : "major",
     "grid.color": "lightgray",
     "grid.linestyle": ":",
     # legend
     "legend.fontsize":8,
    "legend.facecolor":'white',
    "legend.framealpha":1.0 ,  
     })

#%%
if __name__ == '__main__':
        
    #% READ THE DATA AND STORE IN DICTIONARY
    
    # init spectra with scaled dose to water cylinder 
    spec_a = xRaySpectrum(f'{spec_dir}/{spec_name1}_1mGy_float32.bin', spec_name1)
    spec_b = xRaySpectrum(f'{spec_dir}/{spec_name2}_1mGy_float32.bin', spec_name2)
    spec_a.rescale_counts(dose_target / dose_spec)  
    spec_b.rescale_counts(dose_target / dose_spec)  # could also use `imaging_system.get_water_dose()`
    
    
    # # init detector 
    # detector = Detector(detector_filename, detector_mode, eta=detector_eta)
    
    
    # Vanilla PCD : SNR vs. r for each t_bone (one spectral combo)
    
    specs_id = f'{spec_a.name}_{spec_b.name}'
    data_mat1_pcd = np.zeros([len(t_bones), len(r_vec)], dtype=np.float64)
    data_mat2_pcd = np.zeros([len(t_bones), len(r_vec)], dtype=np.float64)
    for i, t_2 in enumerate(t_bones):
        fname1 = outd_pcd + specs_id + f'/mat1_{int(t_1):02}tiss_{int(t_2):02}bone.bin'
        fname2 = outd_pcd + specs_id + f'/mat2_{int(t_1):02}tiss_{int(t_2):02}bone.bin'
        data_mat1_pcd[i,:] = np.fromfile(fname1, dtype=np.float64)
        data_mat2_pcd[i,:] = np.fromfile(fname2, dtype=np.float64)
    
    data_mat1_eid = np.zeros([len(t_bones), len(r_vec)], dtype=np.float64)
    data_mat2_eid = np.zeros([len(t_bones), len(r_vec)], dtype=np.float64)
    for i, t_2 in enumerate(t_bones):
        fname1 = outd_eid + specs_id + f'/mat1_{int(t_1):02}tiss_{int(t_2):02}bone.bin'
        fname2 = outd_eid + specs_id + f'/mat2_{int(t_1):02}tiss_{int(t_2):02}bone.bin'
        data_mat1_eid[i,:] = np.fromfile(fname1, dtype=np.float64)
        data_mat2_eid[i,:] = np.fromfile(fname2, dtype=np.float64)
    
    
    
    # Spectral PCD : SNR vs. r for each t_bone (one spectrum, many energy thresholds)
    for E_thresh in E_thresh_vec:        
        spec = spec_a  # use the first spec [MV] only
        #detector_low = Detector(detector_filename, detector_mode, eta=detector_eta, E_threshold_high=E_thresh)
        #detector_high = Detector(detector_filename, detector_mode, eta=detector_eta, E_threshold_low=E_thresh)
    
        outd_j = outd_pcd + f'{spec_a.name}_spectral_{E_thresh:03}keV/'
        
    
    # SNR vs. r for each t_bone
    spectral_data_mat1 = {}
    spectral_data_mat2 = {}
    
    for E_thresh in E_thresh_vec:
        spectral_id = f'{spec_a.name}_spectral_{E_thresh:03}keV/'
        data_array_mat1 = np.zeros([len(t_bones), len(r_vec)], dtype=np.float64)
        data_array_mat2 = np.zeros([len(t_bones), len(r_vec)], dtype=np.float64)
        for i, t_2 in enumerate(t_bones):
            fname1 = outd_pcd + spectral_id + f'mat1_{int(t_1):02}tiss_{int(t_2):02}bone.bin'
            fname2 = outd_pcd + spectral_id + f'mat2_{int(t_1):02}tiss_{int(t_2):02}bone.bin'
            data_array_mat1[i,:] = np.nan_to_num(np.fromfile(fname1, dtype=np.float64))
            data_array_mat2[i,:] = np.nan_to_num(np.fromfile(fname2, dtype=np.float64))
        spectral_data_mat1[E_thresh] = data_array_mat1
        spectral_data_mat2[E_thresh] = data_array_mat2
    
    
    
    # #%% Make some plots
    # #snr1_heatmap = np.zeros([E_thresh_vec.shape])
    # snr1_opt_E_opt_r = []
    # for E_thresh in E_thresh_vec:
    #     snr1_opt_E = np.max(spectral_data_mat1[E_thresh], axis=0)
    #     snr1_opt_E_opt_r.append(np.max(snr1_opt_E))
    
    # fig, ax = plt.subplots(1, 1)
    # #     ax.plot(r_vec, snr1_opt_E)
    # ax.plot(E_thresh_vec, snr1_opt_E_opt_r, 'k.')
    # fig.tight_layout()
    # plt.show()
    # #for E_thresh in E_thresh_vec:
        
    
    #%%  EID vs PCD (MV-kV) - SNR heatmaps t_bone vs r
    dbone = t_bones[1] - t_bones[0]
    matnames = ['Tissue', 'Bone']
    matkw = {'Tissue': {'vmax':70, 'vmin':5}, 'Bone': {'vmax':13, 'vmin':0.5}}
    for mode, datas in [['PCD', [data_mat1_pcd, data_mat2_pcd]], 
                        ['EID', [data_mat1_eid, data_mat2_eid]]]: 
        fig, ax = plt.subplots(1, 2, figsize=[7,3])
        for i, data_i in enumerate(datas):
            m = ax[i].imshow(data_i.T, cmap='coolwarm', aspect='auto', 
                    extent=(min(t_bones)-dbone/2, max(t_bones)+dbone/2, min(r_vec), max(r_vec)),
                    **matkw[matnames[i]])
            fig.colorbar(m, ax=ax[i])
            ax[i].set_xticks(t_bones)
            ax[i].set_xlabel('bone thickness [cm]')
            ax[i].set_ylabel('dose allocation')
            ax[i].set_title(f'{matnames[i]} SNR')   
        fig.tight_layout()
        fig.suptitle(f'MV-kV dual energy {mode}')
        plt.show()
    
    #%% Spectral PCD (MV) - SNR(r_opt) heatmaps t_bone vs E_thresh

    spectral_heatmap_mat1 = np.zeros([len(t_bones), len(E_thresh_vec)])
    spectral_heatmap_mat2 = np.zeros([len(t_bones), len(E_thresh_vec)])
    for i, E_thresh in enumerate(E_thresh_vec):
        spectral_heatmap_mat1[:,i] = np.max(spectral_data_mat1[E_thresh], axis=1)
        spectral_heatmap_mat2[:,i] = np.max(spectral_data_mat2[E_thresh], axis=1)
    
    datas = [spectral_heatmap_mat1, spectral_heatmap_mat2]
    matnames = ['Tissue', 'Bone']
    fig, ax = plt.subplots(1, 2, figsize=[7,3])
    for i, data_i in enumerate(datas):
        m = ax[i].imshow(data_i.T, cmap='coolwarm', aspect='auto', origin='lower',
                extent=(min(t_bones)-dbone/2, max(t_bones)+dbone/2, min(E_thresh_vec), max(E_thresh_vec))
                )
        fig.colorbar(m, ax=ax[i])
        ax[i].set_xticks(t_bones)
        ax[i].set_xlabel('bone thickness [cm]')
        ax[i].set_ylabel('energy threshold [keV]')
        ax[i].set_title(f'{matnames[i]}' + ' SNR($r_{opt}$)') 
    fig.tight_layout()
    fig.suptitle('Spectral PCD (MV only)')
    plt.show()
            
    
    #%% Dual-energy PCD and EID - SNR(r_opt) vs t_bone
    modekw = {'PCD': {'label':'PCD', 'color':'r', 'linestyle':'-' },
              'EID': {'label':'EID', 'color':'b', 'linestyle':'--' }}
    
    fig, ax = plt.subplots(1, 2, figsize=[7,3]) 
    for mode, datas in [['PCD', [data_mat1_pcd, data_mat2_pcd]], 
                        ['EID', [data_mat1_eid, data_mat2_eid]]]: 
        ax[0].plot(t_bones, np.max(datas[0], axis=1), **modekw[mode])
        ax[1].plot(t_bones, np.max(datas[1], axis=1), **modekw[mode])
    for i in range(2): 
        ax[i].set_xlabel('bone thickness [cm]')
        ax[i].set_ylabel('SNR($r_{opt}$)')
        ax[i].legend()
        ax[i].set_title(matnames[i])
    fig.tight_layout()
    fig.suptitle('Dual energy: SNR($r_{opt}$)')
    plt.show()















    
    
    
    
    
    
    
    


