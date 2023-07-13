 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 10:31:34 2023

@author: giajadick

This file can be used to make plots comparing EID and PCD results.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from main import r_vec, t_1, t_bones   # get some params

# label the run according to the params
dose_target = 1e-6        # [Gy]
ideal_detector = False    # T/F, detective efficiency = 1 ?
run_id = f'compare_EID_PCD_{int(1e6*dose_target):04}uGy'
eid_dir = f'output/EID_{int(1e6*dose_target):04}uGy'
pcd_dir = f'output/PCD_{int(1e6*dose_target):04}uGy'

if ideal_detector:
   run_id = run_id + '_ideal'
   eid_dir = eid_dir + '_ideal'
   pcd_dir = pcd_dir + '_ideal'

figd = f'output/figs/{run_id}/'
savefig = False
if savefig:
   os.makedirs(figd, exist_ok=True)
  
  
#% figure parameters

plt.rcParams.update({
   # figure
   "figure.dpi": 600,
   # text
   "font.size":10,
   #"font.family": "serif",
   #"font.serif": ['Computer Modern Roman'],
   #"text.usetex": True,
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
  # "legend.framealpha":1.0 , 
    })

def bf(string): 
   '''make text boldface'''
   #return "\\textbf{"+string+"}"
   return string # does nothing

def label_panels(ax, c='k', loc='outside', dx=-0.06, dy=0.06, fontsize=None,
                label_type='lowercase', label_format='({})'):
   '''
   Function to label panels of multiple subplots in a single figure.
   Parameters
   ----------
   ax : matplotlib AxesSubplot
   c : (str) color of text. The default is 'k'.
   loc : (str), location of label, 'inside' or 'outside'.
   dx : (float) x location relative to upper left corner. The default is 0.07.
   dy : (float) y location relative to upper left corner. The default is 0.07.
   fontsize : (number), font size of label. The default is None.
   label_type : (str), style of labels. The default is 'lowercase'.
   label_format : (str) format string for label. The default is '({})'.

   '''
   if 'upper' in label_type:
       labels = list(map(chr, range(65,91)))
   elif 'lower' in label_type:
       labels = list(map(chr, range(97, 123)))
   else: # default to numbers
       labels = np.arange(1,27).astype(str)
   labels = [ label_format.format(x) for x in labels ]
   # get location of text
loc = 'outside'
dx = -0.06
dy = 0.06
axi = 1

if loc == 'outside':
     xp, yp = -dx, 1+dy
else:
    xp, yp = dx, 1-dy

xloc = 1 + (100-1)*xp
yloc = 1 + (100-1)*yp

number = 100
text = str(number) 
#label = label_panels[1]
#axi.text(xloc, yloc, color= 'blue', fontsize=8,
#va='center', ha='center')
###################### READ THE DATA AND STORE IN DICTIONARY
#
# This section reads the data for EID and PCD runs into
# different dictionaries for each material.
#
# For a given DE spectral pair (e.g. `spec_id = '140kV_80kV'`), the
# corresponding dictionary entry is a 2D array. Each row is an SNR(r)
# curve corresponding to a different bone thickness.
# The arrays of r and t_bone values are stored in `r_vec` and `t_bones`.
#
# There is a dictionary for each basis material (mat1 = tissue, mat2 = bone).
#
# For example, to get the tissue basis material SNR as a function of
# dose allocation r when imaging a bone thickness of 4 cm using EID:
# ```
# cm_bone = 4
# spec_id = '140kV_80kV'
# SNR_tissue_4cm = data_eid_mat1[spec_id][cm_bone - 1]  # subtract 1 to index from 0!
# ```
#
# - GJ
#

def fname(dirname, specs, t_1, t_2, mat):
   '''Shorthand for getting data filenames'''
   return os.path.join(dirname, specs) + f'/mat{int(mat)}_{int(t_1):02}tiss_{int(t_2):02}bone.npy'

# SNR vs. r for each t_bone
data_eid_mat1 = {}
data_eid_mat2 = {}
data_pcd_mat1 = {}
data_pcd_mat2 = {}
spec_savelist = ['6MV Treatment', 'MV Detuned', '140kV', '120kV', '80kV']
# iterate over all spectral combinations
# if specs array is ordered, then spec_a is always high energy
for j, spec1 in enumerate(spec_savelist[:-1]):
   for spec2 in spec_savelist[j+1:]:
      
      specs_id = f'{spec1}_{spec2}' 
      array_eid_mat1 = np.zeros([len(t_bones), len(r_vec)],
1e6*dose_target)

array_eid_mat1 = np.zeros([len(t_bones), len(r_vec)], dtype=np.float64)
array_eid_mat2 = np.zeros([len(t_bones), len(r_vec)], dtype=np.float64)
array_pcd_mat1 = np.zeros([len(t_bones), len(r_vec)], dtype=np.float64)
array_pcd_mat2 = np.zeros([len(t_bones), len(r_vec)], dtype=np.float64)

      
for i, t_2 in enumerate(t_bones):
           array_eid_mat1[i,:] = np.fromfile(fname(eid_dir, specs_id, t_1, t_2, 1), dtype=np.float64)
           array_eid_mat2[i,:] = np.fromfile(fname(eid_dir, specs_id, t_1, t_2, 2), dtype=np.float64)
           array_pcd_mat1[i,:] = np.fromfile(fname(pcd_dir, specs_id, t_1, t_2, 1), dtype=np.float64)
           array_pcd_mat2[i,:] = np.fromfile(fname(pcd_dir, specs_id, t_1, t_2, 2), dtype=np.float64)
        
data_eid_mat1[specs_id] = array_eid_mat1
data_eid_mat2[specs_id] = array_eid_mat2
data_pcd_mat1[specs_id] = array_pcd_mat1
data_pcd_mat2[specs_id] = array_pcd_mat2

# dect labels
dect_id_mvkv = 'MV Detuned_80kV'

####################### EXAMPLE PLOT - SNR vs. r for EID and PCD
#
# To make this snippet, I mostly copy-pasted the structure of plot 3
# "PLOT 3 :  DOSE DISTRIBUTION for x-cm bone" from the plots.py script
# and modified it so that:
#
#   i. The only spec pair is detunedMV-80kV, (These plots should focus on
#      comparing EID with PCD, whereas before we compared MV-kV with kV-kV)
#
#  ii. There are three panels, and each is a different bone thickness.
#      (Before, the bone thicknesses were plotted on the same panel,
#      and each panel was a different spec pair)
#
# iii. Within each panel, there are two curves, for EID and PCD.
# 
# There are still two separate figures produced, one for each basis material.
#
# - GJ
#

# choose the spectral pair
spec_id = 'MV Detuned_80kV'
spec_title = 'detunedMV-80kV'

# assign some plotting params
color_eid = 'red'
color_pcd = 'blue'
line_eid = '-'       # solid
line_pcd = '--'      # dashed
share_y_axis = True  # set all subplot y-axes to have same min and max?

# iterate over the two basis materials
for i, data_mat in enumerate([[data_eid_mat1, data_pcd_mat1],
                             [data_eid_mat2, data_pcd_mat2]]):
   # unpack the data
   data_eid, data_pcd = data_mat
   mat_id = ['tissue','bone'][i]
  
   # choose the bone thicknesses to plot
   t_bones_plot = [2,4,6,8]#[1, 4, 6]
   N = len(t_bones_plot)
  
   fig, ax = plt.subplots(1,N, figsize=[3*N+1,3], sharey=share_y_axis)
   if type(ax) != np.ndarray: # for N = 1
       ax = np.array([2,4,6,8], [10,12,14]) #change made, [ax]
   # plot the data and print coordinates of peak
   print('\n\t\t\t\t\t EID \t\t\t\t PCD')
   print('mat  \t t_bone \t\t r_max \t SNR_max \t r_max \t SNR_max')
   for j, cm in enumerate(t_bones_plot):   # index = 1 - cm
       snr_eid = data_eid[spec_id][cm-1]
       snr_pcd = data_pcd[spec_id][cm-1]
       rmax_eid = r_vec[np.argmax(snr_eid)]
       rmax_pcd = r_vec[np.argmax(snr_pcd)]
       print(f'{mat_id} \t {cm:.2f} \t\t {rmax_eid:.2f} \t {np.max(snr_eid):.2f} \t\t {rmax_pcd:.2f} \t {np.max(snr_pcd):.2f}')

       ax[j].set_title(f'{cm} cm bone')
       ax[j].plot(r_vec, snr_eid, color=color_eid, ls=line_eid, label='EID')
       ax[j].plot(r_vec, snr_pcd, color=color_pcd, ls=line_pcd, label='PCD')
      
       ax[j].axvline(rmax_eid, color=color_eid, ls=line_eid)
       ax[j].axvline(rmax_pcd, color=color_pcd, ls=line_pcd)
   # some extra plot formatting
   x = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
   y = [0, 30, 60, 90, 120, 150]

   plt.plot(x, y)
   plt.xlabel('dose to spec1')
   plt.ylabel('SNR')

   fig.suptitle(f'{spec_title}, {mat_id} SNR', y=0.92)  # default y=0.98
   fig.supylabel(f'{mat_id} SNR')
   ax[0].legend(loc='upper left')
   for axi in ax:
       axi.set_ylim(0, None)
       axi.set_xlim(0,1)
       axi.set_xlabel('dose to spec1')
       axi.set_ylabel('SNR')
   label_panels(ax)
   fig.tight_layout()
   if savefig:
       plt.savefig(figd+f'plot3_snr_v_dose_multi_mat{i+1}.pdf')
   plt.show()
