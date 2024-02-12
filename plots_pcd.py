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
from scipy import interpolate

# define / load the parameters to use 
from main import r_vec, t_bones, t_tissues, dose_spec, spec_dir  # , t_1

matnames = ['Tissue', 'Bone']

#from plots import bf, label_panels, heatmap
dose_target = 1e-6  
detector_std_e = 10       # electronic noise, only used for EID mode

# define data directories
spec_name1 = 'detunedMV' 
#spec_name1 = '140kV'
spec_name2 = '80kV'
figd = f'output/figs/detcomp_{spec_name1}_{spec_name2}/'
outd_pcd = f'output/PCD_{int(1e6*dose_target):04}uGy/'
outd_eid =  f'output/EID_{int(detector_std_e):04}std_{int(1e6*dose_target):04}uGy/' 


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


# #%%
if __name__ == '__main__':
        
    #% READ THE DATA AND STORE IN DICTIONARY

    
    # init spectra with scaled dose to water cylinder 
    spec_a = xRaySpectrum(f'{spec_dir}/{spec_name1}_1mGy_float32.bin', spec_name1)
    spec_b = xRaySpectrum(f'{spec_dir}/{spec_name2}_1mGy_float32.bin', spec_name2)
    spec_a.rescale_counts(dose_target / dose_spec)  
    spec_b.rescale_counts(dose_target / dose_spec)  # could also use `imaging_system.get_water_dose()`
    

    # PCD + EID: SNR vs. r for each t_bone (one spectral combo)
    specs_id = f'{spec_a.name}_{spec_b.name}'
    dshape = [len(t_tissues), len(t_bones), len(r_vec)]
    data_mat1_pcd = np.zeros(dshape, dtype=np.float64)
    data_mat2_pcd = np.zeros(dshape, dtype=np.float64)
    data_mat1_eid = np.zeros(dshape, dtype=np.float64)
    data_mat2_eid = np.zeros(dshape, dtype=np.float64)
    
    for j, t_1 in enumerate(t_tissues):
        for i, t_2 in enumerate(t_bones):
            mat_id = f'{t_1:04.1f}tiss_{t_2:04.1f}bone'

            fname1 = outd_pcd + specs_id + f'/mat1_{mat_id}.bin'
            fname2 = outd_pcd + specs_id + f'/mat2_{mat_id}.bin'
            data_mat1_pcd[j, i,:] = np.fromfile(fname1, dtype=np.float64)
            data_mat2_pcd[j, i,:] = np.fromfile(fname2, dtype=np.float64)
        
            fname1 = outd_eid + specs_id + f'/mat1_{mat_id}.bin'
            fname2 = outd_eid + specs_id + f'/mat2_{mat_id}.bin'
            data_mat1_eid[j, i,:] = np.fromfile(fname1, dtype=np.float64)
            data_mat2_eid[j, i,:] = np.fromfile(fname2, dtype=np.float64)

    # # for ONE tissue thickness
    # 
    # # PCD : SNR vs. r for each t_bone (one spectral combo)
    # specs_id = f'{spec_a.name}_{spec_b.name}'
    # data_mat1_pcd = np.zeros([len(t_bones), len(r_vec)], dtype=np.float64)
    # data_mat2_pcd = np.zeros([len(t_bones), len(r_vec)], dtype=np.float64)
    # for i, t_2 in enumerate(t_bones):
    #     fname1 = outd_pcd + specs_id + f'/mat1_{int(t_1):02}tiss_{t_2:04.1f}bone.bin'
    #     fname2 = outd_pcd + specs_id + f'/mat2_{int(t_1):02}tiss_{t_2:04.1f}bone.bin'
    #     data_mat1_pcd[i,:] = np.fromfile(fname1, dtype=np.float64)
    #     data_mat2_pcd[i,:] = np.fromfile(fname2, dtype=np.float64)
    #
    # # EID : with and without electronic noise
    # data_mat1_eid = np.zeros([len(t_bones), len(r_vec)], dtype=np.float64)
    # data_mat2_eid = np.zeros([len(t_bones), len(r_vec)], dtype=np.float64)
    # for i, t_2 in enumerate(t_bones):
    #     fname1 = outd_eid + specs_id + f'/mat1_{int(t_1):02}tiss_{t_2:04.1f}bone.bin'
    #     fname2 = outd_eid + specs_id + f'/mat2_{int(t_1):02}tiss_{t_2:04.1f}bone.bin'
    #     data_mat1_eid[i,:] = np.fromfile(fname1, dtype=np.float64)
    #     data_mat2_eid[i,:] = np.fromfile(fname2, dtype=np.float64)

# # %% some calcs on the data
# heatmaps of the peak coordinates (r, SNR) for each (t_tissue, t_bone) pair.
    
    snrmax_mat1_pcd = np.max(data_mat1_pcd, axis=2)
    snrmax_mat2_pcd = np.max(data_mat2_pcd, axis=2)
    snrmax_mat1_eid = np.max(data_mat1_eid, axis=2)
    snrmax_mat2_eid = np.max(data_mat2_eid, axis=2)
    snrmax_mat1_pcd.ravel().tofile(outd_pcd + 'snrmax_mat1.bin')
    snrmax_mat2_pcd.ravel().tofile(outd_pcd + 'snrmax_mat2.bin')
    snrmax_mat1_eid.ravel().tofile(outd_eid + 'snrmax_mat1.bin')
    snrmax_mat2_eid.ravel().tofile(outd_eid + 'snrmax_mat2.bin')
    
    rmax_mat1_pcd = r_vec[np.argmax(data_mat1_pcd, axis=2)]
    rmax_mat2_pcd = r_vec[np.argmax(data_mat2_pcd, axis=2)]
    rmax_mat1_eid = r_vec[np.argmax(data_mat1_eid, axis=2)]
    rmax_mat2_eid = r_vec[np.argmax(data_mat2_eid, axis=2)]
    rmax_mat1_pcd.ravel().tofile(outd_pcd + 'rmax_mat1.bin')
    rmax_mat2_pcd.ravel().tofile(outd_pcd + 'rmax_mat2.bin')
    rmax_mat1_eid.ravel().tofile(outd_eid + 'rmax_mat1.bin')
    rmax_mat2_eid.ravel().tofile(outd_eid + 'rmax_mat2.bin')


#%% Dual-energy PCD and EID - SNR(r_opt) vs t_bone
    
    modekw = {'PCD': {'label':'PCD', 'color':'b', 'linestyle':'-' },
              'EID': {'label':'EID', 'color':'r', 'linestyle':'--' }
             }
    
    fig, ax = plt.subplots(1, 2, figsize=[7,3]) 
    for mode, datas in [['PCD', [data_mat1_pcd, data_mat2_pcd]], 
                        ['EID', [data_mat1_eid, data_mat2_eid]],
                       ]: 
        ax[0].plot(t_bones, np.max(datas[0], axis=1), **modekw[mode])
        ax[1].plot(t_bones, np.max(datas[1], axis=1), **modekw[mode])
    for i in range(2): 
        ax[i].set_xlabel('bone thickness [cm]')
        ax[i].set_ylabel('SNR($r_{opt}$)')
        ax[i].legend()
        ax[i].set_title(matnames[i])
    fig.tight_layout()
    if savefig:
        plt.savefig(f'{figd}/SNRopt_{mode}.png', bbox_inches='tight')
    else:
        fig.suptitle(f'{spec_a.name}-{spec_b.name} - {mode}', y=1.02)
    plt.show()



#%% FIG : optimal dose vs t_bone

    modekw = {'PCD': {'label':'PCD', 'color':'b', 'linestyle':'-' },
              'EID': {'label':'EID', 'color':'r', 'linestyle':'--' }
             }
    
    fig, ax = plt.subplots(1, 2, figsize=[7,3]) 
    for mode, datas in [['PCD', [data_mat1_pcd, data_mat2_pcd]], 
                        ['EID', [data_mat1_eid, data_mat2_eid]],
                       ]: 
        ax[0].plot(t_bones, r_vec[np.argmax(datas[0], axis=1)], **modekw[mode])
        ax[1].plot(t_bones, r_vec[np.argmax(datas[1], axis=1)], **modekw[mode])
    for i in range(2): 
        ax[i].set_xlabel('bone thickness [cm]')
        ax[i].set_ylabel('$r_{opt}$')
        ax[i].legend()
        ax[i].set_title(matnames[i])
    fig.tight_layout()
    #if savefig:
    #    plt.savefig(f'{figd}/r_opt_{mode}.png', bbox_inches='tight')
    #else:
    #    fig.suptitle(f'{spec_a.name}-{spec_b.name}, {int(dose_target/1e-6)}'+' $\mu$Gy: SNR($r_{opt}$)', y=1.02)
    plt.show()



#%%  EID vs PCD (MV-kV) - SNR heatmaps t_bone vs r
    
    dbone = t_bones[1] - t_bones[0]
    matkw = {'Tissue': {'vmax':100, 'vmin':0}, 'Bone': {'vmax':20, 'vmin':0}}  # 1 uGy + elec noise
    # matkw = {'Tissue': {}, 'Bone': {}}  # temp, for choosing kwargs
    matnames = list(matkw.keys())
    for mode, datas in [['PCD', [data_mat1_pcd, data_mat2_pcd]], 
                        ['EID', [data_mat1_eid, data_mat2_eid]]]: 
        fig, ax = plt.subplots(1, 2, figsize=[7,3])
        for i, data_i in enumerate(datas):
            m = ax[i].imshow(data_i.T, cmap='inferno', aspect='auto', origin='lower',
                    extent=(min(t_bones), max(t_bones), min(r_vec), max(r_vec)),
                    **matkw[matnames[i]])
            cb = fig.colorbar(m, ax=ax[i])
            r_opt_i = r_vec[np.argmax(data_i, axis=1)]
            ax[i].plot(t_bones, r_opt_i, 'k-', lw=2)
            ax[i].plot(t_bones, r_opt_i, 'w-', lw=1)
            ax[i].set_xticks(np.arange(1,11,1))
            ax[i].set_xlabel('bone thickness [cm]')
            ax[i].set_ylabel('MV dose allocation')
            ax[i].set_title(f'{matnames[i]} SNR')   
        cb.set_ticks(np.arange(0, 21, 4))  # cheesy change to 2nd cbar ticks
        fig.tight_layout()
        if savefig:
            plt.savefig(f'{figd}/heatmap_{mode}.png', bbox_inches='tight')
        else:
            fig.suptitle(f'{spec_a.name}-{spec_b.name} - {mode}')
        plt.show()
    

    
#%% FIG : heatmap of SNR_peak for t_1, t_2
    matkw = {'Tissue': {'vmin':0, 'vmax':110}, 
             'Bone': {'vmin':0, 'vmax':110}}  
    #matkw = {'Tissue': {}, 'Bone': {}}  # temp, for choosing kwargs
    matnames = list(matkw.keys())
    for mode, datas in [['PCD', [snrmax_mat1_pcd, snrmax_mat2_pcd]], 
                        ['EID', [snrmax_mat1_eid, snrmax_mat2_eid]]]: 
        fig, ax = plt.subplots(1, 2, figsize=[7,3])
        for i, data_i in enumerate(datas):
            m = ax[i].imshow(data_i.T, cmap='inferno', aspect='auto', origin='lower',
                    extent=(min(t_tissues ), max(t_tissues), min(t_bones), max(t_bones)),
                    **matkw[matnames[i]])
            cb = fig.colorbar(m, ax=ax[i])
            ax[i].set_xlabel('tissue thickness [cm]')
            ax[i].set_ylabel('bone thickness [cm]')
            ax[i].set_title(f'{matnames[i]}'+' SNR$_{max}$')   
        #cb.set_ticks(np.arange(0, 21, 4))  # cheesy change to 2nd cbar ticks
        fig.tight_layout()
        if savefig:
            plt.savefig(f'{figd}/snrmax_{mode}.png', bbox_inches='tight')
        else:
            fig.suptitle(f'{spec_a.name}-{spec_b.name} - {mode}')
        plt.show()
    

    #%% FIG : heatmap of SNR_peak for t_1, t_2
        #matkw = {'Tissue': {'vmin':0, 'vmax':110}, 
        #         'Bone': {'vmin':0, 'vmax':110}}  
        matkw = {'Tissue': {'vmin':0.1, 'vmax':0.9}, 'Bone': {'vmin':0.1, 'vmax':0.9}}  # temp, for choosing kwargs
        matnames = list(matkw.keys())
        for mode, datas in [['PCD', [rmax_mat1_pcd, rmax_mat2_pcd]], 
                            ['EID', [rmax_mat1_eid, rmax_mat2_eid]]]: 
            fig, ax = plt.subplots(1, 2, figsize=[7,3])
            for i, data_i in enumerate(datas):
                m = ax[i].imshow(data_i.T, cmap='bwr', aspect='auto', origin='lower',
                        extent=(min(t_tissues ), max(t_tissues), min(t_bones), max(t_bones)),
                        **matkw[matnames[i]])
                cb = fig.colorbar(m, ax=ax[i])
                ax[i].set_xlabel('tissue thickness [cm]')
                ax[i].set_ylabel('bone thickness [cm]')
                ax[i].set_title(f'{matnames[i]}'+' r(SNR$_{max}$)')   
            #cb.set_ticks(np.arange(0, 21, 4))  # cheesy change to 2nd cbar ticks
            fig.tight_layout()
            if savefig:
                plt.savefig(f'{figd}/rmax_{mode}.png', bbox_inches='tight')
            else:
                fig.suptitle(f'{spec_a.name}-{spec_b.name} - {mode}')
            plt.show()
        

        
    


# #%% compute the r vector for given t1, t2 vector
#     profdir = '../dex-ct-sim/output/mvkv_pcd/IQ7_PCD/'
#     Nkern = 15
#     t1profile = np.fromfile(profdir + f't1profile_kern{Nkern:02}_float32.bin', dtype=np.float32)
#     t2profile = np.fromfile(profdir + f't2profile_kern{Nkern:02}_float32.bin', dtype=np.float32)

#     # fig, ax = plt.subplots(1, 2, figsize=[7,3])
#     # ax[0].plot(t1profile)
#     # ax[1].plot(t2profile)
#     # fig.tight_layout()
#     # plt.show()

#     t1grid, t2grid = np.meshgrid(t_tissues, t_bones)
#     #f_rmax_mat1 = interpolate.interp2d(t1grid, t2grid, rmax_mat1_pcd)
#     f_rmax_mat1 = interpolate.RectBivariateSpline(t1grid, t2grid, rmax_mat1_pcd)
#     rmax_data = f_rmax_mat1(t1profile/p_1, t2profile/p_2)
    
#     #plt.plot(rmax_data)
#     #plt.show()
#     print(rmax_data.shape)



