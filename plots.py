#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 14:54:31 2023

@author: giajadick
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from imaging_system import Source, Material, Detector
from main import dose_target, run_id, outd, r_vec, t_1, t_bones, detector_mode, ideal_detector  # get some params 

figd = f'output/figs/{run_id}/'
savefig = True
if savefig:
    os.makedirs(figd, exist_ok=True)


#%% figure parameters

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

###################### READ THE DATA AND STORE IN DICTIONARY

# spectra files
spec_dir = './input/spectrum/'
spec_files, spec_names = np.array([
            ['Accuray_treatment6MV.csv', '6MV Treatment' ],
            ['Accuray_detuned.csv',      'MV Detuned'    ],
            ['spec140.mat',              '140kV'         ],
            ['spec120.mat',              '120kV'         ],
            ['spec80.mat',               '80kV'          ],
            ]).T
# load the 5 source spectra and rescale to target dose
specs = []
water = Material('water', 1.0, 20.0)  # center of 40 cm water cylinder
for j in range(len(spec_files)):
    spec_j = Source(spec_dir+spec_files[j], spec_names[j])
    scale = dose_target / spec_j.get_water_dose(water)        
    spec_j.rescale_I0(scale)
    specs.append(spec_j)
    
# detector 
detector_filename = './input/detector/eta.npy'   
detector = Detector(detector_filename, detector_mode, ideal_detector)

    
# SNR vs. r for each t_bone
data_mat1 = {}
data_mat2 = {}
for spec1 in spec_names:
    for spec2 in spec_names:
        
        specs_id = f'{spec1}_{spec2}'
        data_array_mat1 = np.zeros([len(t_bones), len(r_vec)], dtype=np.float64)
        data_array_mat2 = np.zeros([len(t_bones), len(r_vec)], dtype=np.float64)
        
        for i, t_2 in enumerate(t_bones):
            fname1 = outd + specs_id + f'/mat1_{int(t_1):02}tiss_{int(t_2):02}bone.npy'
            fname2 = outd + specs_id + f'/mat2_{int(t_1):02}tiss_{int(t_2):02}bone.npy'
        
            data_array_mat1[i,:] = np.fromfile(fname1, dtype=np.float64)
            data_array_mat2[i,:] = np.fromfile(fname2, dtype=np.float64)
        
        data_mat1[specs_id] = data_array_mat1
        data_mat2[specs_id] = data_array_mat2

# dect labels
dect_id1 = 'MV Detuned_80kV'
dect_id2 = '140kV_80kV'

dect_specs = [ 
  '6MV Treatment_140kV',
  '6MV Treatment_120kV',
  '6MV Treatment_80kV',
  'MV Detuned_140kV',
  'MV Detuned_120kV',
  'MV Detuned_80kV',
  '140kV_120kV',
  '140kV_80kV',
  '120kV_80kV',]

dect_spec_names = [x.replace('_', '-') for x in dect_specs]


#%% Some helpful functions for plotting

def bf(string):  
    '''make text boldface'''
    return "\\textbf{"+string+"}"


def label_panels(ax, c='k', loc='inside', dx=0.07, dy=0.07, fontsize=None,
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
    if loc == 'outside':
        xp, yp = -dx, 1+dy
    else:
        xp, yp = dx, 1-dy
        
    for i, axi in enumerate(ax.ravel()):
        xmin, xmax = axi.get_xlim()
        ymin, ymax = axi.get_ylim()
        xloc = xmin + (xmax-xmin)*xp
        yloc = ymin + (ymax-ymin)*yp
        
        label = labels[i]
        print(xloc, yloc)
        axi.text(xloc, yloc, bf(label), color=c, fontsize=fontsize,
          va='center', ha='center')
        
    return None


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center",
              fontsize=8)
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = mpl.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts



#%% ####################### TABLES: coords of r_peak, SNR_peak
bone_ind = 0 # 1 cm bone

## TWO TABLES
# for i_mat, data_mat in enumerate([data_mat1, data_mat2]):
#     print(f'\n\nTABLE {i_mat+1}\n')
#     for dect_id in specs:
#         spect_name = dect_id.replace('_', '-')
#         v1 = data_mat[dect_id][0]
#         rmax1, snrmax1 = r_vec[np.argmax(v1)], np.max(v1)
#         print(f'{spect_name:20}  & {rmax1:6.2f}  &   {snrmax1:6.2f}    \\\\' )

## ONE TABLE
for dect_id in dect_specs:
    spect_name = dect_id.replace('_', '-')
    
    v1 = data_mat1[dect_id][0]
    rmax1, snrmax1 = r_vec[np.argmax(v1)], np.max(v1)
    
    v2 = data_mat2[dect_id][0]
    rmax2, snrmax2 = r_vec[np.argmax(v2)], np.max(v2)
    print(f'{spect_name:20} & {rmax1:6.2f} & {snrmax1:6.2f} & {rmax2:6.2f} & {snrmax2:6.2f}   \\\\' )
    

#%% ######################### PLOT 1 :  OPTIMAL DOSE as FUNC OF TBONE

rmax_dect1_mat1 = r_vec[np.argmax(data_mat1[dect_id1], axis=1)]
rmax_dect1_mat2 = r_vec[np.argmax(data_mat2[dect_id1], axis=1)]
rmax_dect2_mat1 = r_vec[np.argmax(data_mat1[dect_id2], axis=1)]
rmax_dect2_mat2 = r_vec[np.argmax(data_mat2[dect_id2], axis=1)]

fig, ax = plt.subplots(1,1, figsize=[6,4])
ax.set_xticks(t_bones)
ax.set_ylim(0,1)
ax.set_ylabel('optimal dose to spec1 (spec2 = 80kV)' )
ax.set_xlabel('$t_\mathrm{bone}$ [cm]')
ax.plot(t_bones, rmax_dect1_mat1, color='b', label='detunedMV / tissue')
ax.plot(t_bones, rmax_dect1_mat2, color='r', label='detunedMV / bone')
ax.plot(t_bones, rmax_dect2_mat1, color='b', ls='--', label='140kV / tissue')
ax.plot(t_bones, rmax_dect2_mat2, color='r', ls='--', label='140kV / bone')
ax.legend(title=bf('spec1 / basis material'))
fig.tight_layout()
if savefig:
    plt.savefig(figd+'plot1_dose_v_tbone.pdf')
plt.show()
        
        
#%% ########################## PLOT 2 :  SNRmax as FUNC OF TBONE
color = True

### get data
snrmax_dect1_mat1 = np.max(data_mat1[dect_id1], axis=1)
snrmax_dect1_mat2 = np.max(data_mat2[dect_id1], axis=1)
snrmax_dect2_mat1 = np.max(data_mat1[dect_id2], axis=1)
snrmax_dect2_mat2 = np.max(data_mat2[dect_id2], axis=1)

### plot
fig, ax_left = plt.subplots(1,1, figsize=[6,4])
ax = ax_left.twinx()
fname = 'plot2_snrmax_v_tbone_bw.pdf'
if color:
    fname = fname.replace('bw','color')
    c1, c2 = 'rb'
    for i, spine in enumerate(ax.spines.values()):
        spine.set_edgecolor([c1,c2,'k','k'][i])
else:
    c1, c2 = 'kk'

ax_left.set_xticks(t_bones)
ax_left.set_xlabel('$t_\mathrm{bone}$ [cm]')

# mat 1, tissue
ax_left.set_ylabel('max SNR (tissue)' )
ax_left.plot(t_bones, snrmax_dect2_mat1, c=c1, marker='s', markerfacecolor='None', label='140kV-80kV')
ax_left.plot(t_bones, snrmax_dect1_mat1, c=c1, marker='o', markerfacecolor='None', label='MV-80kV')

# mat 2, bone    
ax.set_ylabel('max SNR (bone)')#, rotation=-90 )
ax.plot(t_bones, snrmax_dect2_mat2, ls=':', c=c2, marker='s', markerfacecolor='None', label='140kV-80kV')
ax.plot(t_bones, snrmax_dect1_mat2, ls=':', c=c2, marker='o', markerfacecolor='None', label='MV-80kV')

ax_left.legend(title=bf('tissue'),loc='lower left')
ax.legend(title=bf('bone'), loc='upper right')

# align ticks
# this is somewhat manual, make sure ylims are integer multiples of Nvals
ax.set_ylim(0, 20)
ax_left.set_ylim(0,160)
Nvals = 5
ax_left.set_yticks(np.linspace(0, ax_left.get_ybound()[1], Nvals))
ax.set_yticks(np.linspace(0, ax.get_ybound()[1], Nvals))

if color:
    ax_left.yaxis.label.set_color(c1)
    ax.yaxis.label.set_color(c2)
    ax_left.tick_params(axis='y', colors=c1)
    ax.tick_params(axis='y', colors=c2)

fig.tight_layout()

if savefig:
    plt.savefig(figd + fname)
plt.show()


#%% ########################### PLOT 3 :  DOSE DISTRIBUTION for x-cm bone

for i, data_mat in enumerate([data_mat1, data_mat2]):
    mat_id = ['tissue','bone'][i]
    col = ['b', 'r'][i]
    fig, ax = plt.subplots(1,2, figsize=[7,3], sharey=False)
    #ax[0].set_title(bf('detunedMV-80kV'))
    #ax[1].set_title(bf('140kV-80kV'))
    ax[0].set_title('detunedMV-80kV')
    ax[1].set_title('140kV-80kV')
    
    ax[0].set_ylabel(f'{mat_id} SNR')

    for j, cm in enumerate([1, 4, 6]):   # index = 1 - cm
        snr_spec1 = data_mat[dect_id1][cm-1]
        snr_spec2 = data_mat[dect_id2][cm-1]
        rmax1 = r_vec[np.argmax(snr_spec1)]
        rmax2 = r_vec[np.argmax(snr_spec2)]
        print(mat_id, cm, rmax1, rmax2)
        
        ls = ['-','--',':'][j]
        
        ax[0].plot(r_vec, snr_spec1, ls, color='k', label=f'{cm} cm')
        ax[1].plot(r_vec, snr_spec2, ls, color='k', label=f'{cm} cm')
        
        ax[0].axvline(rmax1, ls=ls, color='k')
        ax[1].axvline(rmax2, ls=ls, color='k')
        
    ax[1].legend(loc='upper right', title='$t_{bone}$')
        
    for axi in ax:
        axi.set_ylim(0, None)
        axi.set_xlim(0,1)
        axi.set_xlabel('dose to spec1')
    
    label_panels(ax)
    fig.tight_layout()
    if savefig:
        plt.savefig(figd+f'plot3c_snr_v_dose_multi_mat{i+1}.pdf')
    plt.show()


#%% ########################### PLOT 4 :  HEATMAP OF SNR

t_bones_s = np.arange(1,11,1).astype(str)
mat1 = np.zeros([len(dect_specs),len(t_bones_s)])
mat2 = np.zeros([len(dect_specs),len(t_bones_s)])

for i, spec in enumerate(dect_specs):
    mat1[i] = np.max(data_mat1[spec], axis=1)
    mat2[i] = np.max(data_mat2[spec], axis=1)
    
for basismat, mat, kwargs in [['tissue',mat1, {'vmin':0, 'vmax':100}]
                      , ['bone',mat2, {'vmin':0, 'vmax': 14}]]:
    fig, ax = plt.subplots(1, 1, figsize=[6.5,3.8])
    ax.grid(False)
    ax.tick_params(direction="out")
    ax.xaxis.set_ticks_position('top') 
    ax.yaxis.set_ticks_position('left') 
    
    im, cbar = heatmap(mat, dect_spec_names, t_bones_s, ax=ax,
                        cmap="coolwarm", cbarlabel=f"{basismat} SNR", 
                        cbar_kw={'pad':0.02}, **kwargs)
    texts = annotate_heatmap(im, valfmt="{x:.1f}", textcolors=('black','black'))
    ax.set_title('bone thickness [cm]', fontsize=10)
    fig.tight_layout()
    if savefig:
        plt.savefig(figd+f'plot4_heatmap_snr_v_tbone_{basismat}.pdf')
    plt.show()




#%% ###################### PLOT 5 : SPECTRA


lss = ['-','--','-',':','--']
colors = ['k','k','k','k','k']  # black
tlw = 1

fig,ax=plt.subplots(1, 2, figsize=[7, 3])
ax[0].set_ylabel('$1 \\mu Gy$-scaled counts')

ax[0].set_title('kV spectra')
ax[0].set_xlabel('energy [keV]')
ax[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

ax[1].set_title('MV spectra')
ax[1].set_xlim(0,6)
ax[1].set_xlabel('energy [MeV]')


for j,spec in enumerate(specs):
    if 'Accuray' in spec.filename:
        ax[1].plot(spec.E/1000, spec.I0,  lw=tlw, ls=lss[j], color=colors[j], label=spec.name)
    else:
        ax[0].plot(spec.E, spec.I0, lw=tlw, ls=lss[j], color=colors[j], label=spec.name)
        
ax[0].legend(loc='upper right')
ax[1].legend(loc='upper right')
        
label_panels(ax)

fig.tight_layout()
if savefig:
    plt.savefig( figd + 'spectra.pdf' )
plt.show()




#%% ######## Detector efficiency, eta vs. E

fig,ax=plt.subplots(1, 1, figsize=[4, 3])
ax.plot(detector.E, detector.eta, 'k-')
ax.set_xlabel('energy [keV]')
ax.set_ylabel('detective efficiency')
ax.set_ylim(0, None)
fig.tight_layout()
if savefig:
    plt.savefig( figd + 'detector_eta.pdf')
plt.show()














        
        