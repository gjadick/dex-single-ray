#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 14:54:31 2023

@author: giajadick
"""

import sys
sys.path.append('xtomosim')  
from xtomosim.system import xRaySpectrum

# define / load the parameters to use (assumes that detector is not for constant eta)
from main import r_vec, t_1, t_bones, detector, spec_names, spec_dir, dose_target, dose_spec
run_id = 'EID_0001uGy'
outd = f'./output/{run_id}/'

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

figd = f'output/figs/{run_id}/'
figd_r1 = f'output/figs/{detector.mode}/'

savefig = False
if savefig:
    os.makedirs(figd, exist_ok=True)
    os.makedirs(figd_r1, exist_ok=True)
    

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



#%% Some helpful functions for plotting

def bf(string):  
    '''make text boldface'''
    return "\\textbf{"+string+"}"


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


#%%

if __name__ == '__main__':

###################### READ THE DATA AND STORE IN DICTIONARY

    # load spectra
    specs = []
    for name in spec_names:
        file = f'{spec_dir}/{name}_1mGy_float32.bin'  
        spec = xRaySpectrum(file, name)
        spec.rescale_counts(dose_target / dose_spec )
        specs.append(spec)
            
    # SNR vs. r for each t_bone
    data_mat1 = {}
    data_mat2 = {}
    dect_specs = []
    
    for j, spec_a in enumerate(specs[:-1]):
        for jj, spec_b in enumerate(specs[j+1:]):
            specs_id = f'{spec_a.name}_{spec_b.name}'
            if specs_id.count('MV') > 1:  # ignore MV-MV spec combos
                pass
            else:
                dect_specs.append(specs_id)
                data_array_mat1 = np.zeros([len(t_bones), len(r_vec)], dtype=np.float64)
                data_array_mat2 = np.zeros([len(t_bones), len(r_vec)], dtype=np.float64)
                for i, t_2 in enumerate(t_bones):
                    fname1 = outd + specs_id + f'/mat1_{int(t_1):02}tiss_{int(t_2):02}bone.bin'
                    fname2 = outd + specs_id + f'/mat2_{int(t_1):02}tiss_{int(t_2):02}bone.bin'
                    data_array_mat1[i,:] = np.fromfile(fname1, dtype=np.float64)
                    data_array_mat2[i,:] = np.fromfile(fname2, dtype=np.float64)
                data_mat1[specs_id] = data_array_mat1
                data_mat2[specs_id] = data_array_mat2
            
    
    # Two DE-CT acquisitions of interest for comparison: MV-kV versus kV-kV
    dect_id1 = 'detunedMV_80kV'
    dect_id2 = '140kV_80kV'


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
    ax_left.plot(t_bones, snrmax_dect2_mat1, c=c1, marker='s', markerfacecolor='None', label='kV-kV')
    ax_left.plot(t_bones, snrmax_dect1_mat1, c=c1, marker='o', markerfacecolor='None', label='MV-kV')
    
    # mat 2, bone    
    ax.set_ylabel('max SNR (bone)')#, rotation=-90 )
    ax.plot(t_bones, snrmax_dect2_mat2, ls=':', c=c2, marker='s', markerfacecolor='None', label='kV-kV')
    ax.plot(t_bones, snrmax_dect1_mat2, ls=':', c=c2, marker='o', markerfacecolor='None', label='MV-kV')
    
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
        
        im, cbar = heatmap(mat, dect_specs, t_bones_s, ax=ax,
                            cmap="coolwarm", cbarlabel=f"{basismat} SNR", 
                            cbar_kw={'pad':0.02}, **kwargs)
        texts = annotate_heatmap(im, valfmt="{x:.1f}", textcolors=('black','black'))
        ax.set_title('bone thickness [cm]', fontsize=10)
        fig.tight_layout()
        if savefig:
            plt.savefig(figd+f'plot4_heatmap_snr_v_tbone_{basismat}.pdf')
        plt.show()
    



#%% ###################### PLOT 5 : SPECTRA
    
    spec_dir = './input/spectrum/'
    dose_target = 1e-6
    dose_spec = 1e-3
    import sys
    sys.path.append('xtomosim')  # for xtomosim
    from xtomosim.system import xRaySpectrum
    spec_names = [x.name for x in specs]
    
    specs = []
    for name in spec_names:
        file = f'{spec_dir}/{name}_1mGy_float32.bin'  
        spec = xRaySpectrum(file, name)
        spec.rescale_counts(dose_target / dose_spec )
        specs.append(spec)
        
        
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
        if 'MV' in spec.filename:
            ax[1].plot(spec.E/1000, spec.I0,  lw=tlw, ls=lss[j], color=colors[j], label=spec.name)
        else:
            ax[0].plot(spec.E, spec.I0, lw=tlw, ls=lss[j], color=colors[j], label=spec.name)
            
    ax[0].legend(loc='upper right')
    ax[1].legend(loc='upper right')
            
    label_panels(ax, dx=0.05)
    
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
    
    
    
    
    #%% Revision 1 new figure : MV-kV SNRmax vs TBONE with various detector efficiencies 
    
    # # effective energy of one spectrum?
    # dspec = specs[1]  # detunedMV
    # eta_interp = np.interp(dspec.E, detector.E, detector.eta)
    # eta_eff = np.sum(eta_interp*dspec.I0/np.sum(dspec.I0))
    # print(dspec.name, f'effective efficiency = {eta_eff:.5f}')
    
    color = True
    dect_id = 'detunedMV_80kV'  # just one spec combo for this
    eta_vals = [0.05, 0.10, 0.20]
    lss = ['-', '--', ':']
    
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
    
    # ax_left : mat1 (tissue)
    ax_left.set_xticks(t_bones)
    ax_left.set_xlabel('$t_\mathrm{bone}$ [cm]')
    ax_left.set_ylabel('max SNR (tissue)' )
    
    # ax : mat2 (bone)
    ax.set_ylabel('max SNR (bone)')
    
    # plot a line for each constant eta
    for e, eta in enumerate(eta_vals):
        run_id_eta = run_id + f'_{int(100*eta):03}eta'
        outd_eta = f'./output/{run_id_eta}/'
    
        # load the data (not pre-loaded into the dictionaries above!)
        data_array_mat1 = np.zeros([len(t_bones), len(r_vec)], dtype=np.float64)
        data_array_mat2 = np.zeros([len(t_bones), len(r_vec)], dtype=np.float64)
        for i, t_2 in enumerate(t_bones):
            fname1 = outd_eta + dect_id + f'/mat1_{int(t_1):02}tiss_{int(t_2):02}bone.bin'
            fname2 = outd_eta + dect_id + f'/mat2_{int(t_1):02}tiss_{int(t_2):02}bone.bin'
            data_array_mat1[i,:] = np.fromfile(fname1, dtype=np.float64)
            data_array_mat2[i,:] = np.fromfile(fname2, dtype=np.float64)
        snrmax1 = np.max(data_array_mat1, axis=1)
        snrmax2 = np.max(data_array_mat2, axis=1)
        
        # plot the data
        ax_left.plot(t_bones, snrmax1, ls=lss[e], c=c1, marker='', markerfacecolor='None', label=f'{int(100*eta)}\%')
        ax.plot(t_bones, snrmax2,      ls=lss[e], c=c2, marker='', markerfacecolor='None', label=f'{int(100*eta)}\%')
    
    
    # align ticks
    # this is somewhat manual, make sure ylims are integer multiples of Nvals
    ax_left.set_ylim(0,60)
    ax.set_ylim(0, 10)
    Nvals = 6
    ax_left.set_yticks(np.linspace(0, ax_left.get_ybound()[1], Nvals))
    ax.set_yticks(np.linspace(0, ax.get_ybound()[1], Nvals))
    
    if color:
        ax_left.yaxis.label.set_color(c1)
        ax.yaxis.label.set_color(c2)
        ax_left.tick_params(axis='y', colors=c1)
        ax.tick_params(axis='y', colors=c2)
    
    legend_left = ax_left.legend(title=bf('tissue'), loc='upper left', framealpha=1)
    ax.legend(title=bf('bone'), loc='upper right', framealpha=1)
    
    fig.tight_layout()
    if savefig:
        plt.savefig(figd + 'r1_plot1_mvkv_snr_v_bone_etas.pdf')
    plt.show()
    
    
    
    #%%  Revision 1 new figure : optimal dose vs. eta (for given tbone)
    
    t_2 = 1  # single bone thickness
    eta_vals = np.arange(0.01, 1.0, 0.01)
    
    data_array_mat1 = np.zeros([len(eta_vals), len(r_vec)], dtype=np.float64)
    data_array_mat2 = np.zeros([len(eta_vals), len(r_vec)], dtype=np.float64)
    for i, eta in enumerate(eta_vals):
        outd_eta = f'./output/{run_id}_{int(100*eta):03}eta/'
        fname1 = outd_eta + dect_id + f'/mat1_{int(t_1):02}tiss_{int(t_2):02}bone.bin'
        fname2 = outd_eta + dect_id + f'/mat2_{int(t_1):02}tiss_{int(t_2):02}bone.bin'
        data_array_mat1[i,:] = np.fromfile(fname1, dtype=np.float64)
        data_array_mat2[i,:] = np.fromfile(fname2, dtype=np.float64)
    rmax_mat1 = r_vec[np.argmax(data_array_mat1, axis=1)]
    rmax_mat2 = r_vec[np.argmax(data_array_mat2, axis=1)]
    
    fig, ax = plt.subplots(1,1, figsize=[5,3])
    ax.plot(eta_vals, rmax_mat1, color='b', label='tissue')
    ax.plot(eta_vals, rmax_mat2, color='r', label='bone')
    ax.set_ylim(0,1)
    ax.set_ylabel('optimal dose to detunedMV' )
    ax.set_xlabel('detective efficiency $\eta$')
    ax.legend()
    fig.tight_layout()
    if savefig:
        plt.savefig(figd_r1+f'r1_plot2_mvkv_dose_v_eta_{int(t_2):02}bone.pdf')
    plt.show()
            
    
    #%% Revision 1 new figure : SNRmax vs eta for given (t_tiss, t_bone) -- r is optimized
    
    
    
    t_2 = 9 # single bone thickness
    eta_vals = np.arange(0.01, 1.0, 0.01)
    eta_vals = np.delete(eta_vals, 5)  # error here, re run later
    
    data_array_mat1 = np.zeros([len(eta_vals), len(r_vec)], dtype=np.float64)
    data_array_mat2 = np.zeros([len(eta_vals), len(r_vec)], dtype=np.float64)
    for i, eta in enumerate(eta_vals):
        outd_eta = f'./output/{run_id}_{int(100*eta):03}eta/'
        fname1 = outd_eta + dect_id + f'/mat1_{int(t_1):02}tiss_{int(t_2):02}bone.bin'
        fname2 = outd_eta + dect_id + f'/mat2_{int(t_1):02}tiss_{int(t_2):02}bone.bin'
        data_array_mat1[i,:] = np.fromfile(fname1, dtype=np.float64)
        data_array_mat2[i,:] = np.fromfile(fname2, dtype=np.float64)
    SNRmax_mat1 = np.max(data_array_mat1, axis=1)
    SNRmax_mat2 = np.max(data_array_mat2, axis=1)
    
    fig, ax = plt.subplots(1,1, figsize=[5,3])
    # ax.plot(eta_vals, SNRmax_mat1, color='b', label='tissue')
    # ax.plot(eta_vals, SNRmax_mat2, color='r', label='bone')
    #ax.set_ylim(0,1)
    lw = 1.5
    ax.plot(eta_vals, SNRmax_mat1/np.max(SNRmax_mat1), color='b', lw=lw, label=f'tissue, max SNR($\eta$) = {np.max(SNRmax_mat1):.1f}')
    ax.plot(eta_vals, SNRmax_mat2/np.max(SNRmax_mat2), 'r--', lw=lw+0.5, label=f'bone, max SNR($\eta$) = {np.max(SNRmax_mat2):.1f}')
    ax.plot(eta_vals, np.sqrt(eta_vals), 'k:', lw=lw+1, label='theoretical SNR($\eta$) = $\sqrt{\eta}$')
    
    ax.set_ylabel('Normalized dose-optimized SNR($\eta$)' )
    ax.set_xlabel('detective efficiency $\eta$')
    ax.legend()
    fig.tight_layout()
    #if savefig:
    plt.savefig(figd_r1+f'r1_plot2_mvkv_SNR_v_eta_{int(t_2):02}bone.pdf')
    plt.show()
            
    
            
            
    #%% Revision 1 new figure : SNR vs eta for given (t_tiss, t_bone, r)
    
    t_2 = 1  # single bone thickness
    r_vals = [0.5, 0.7, 0.9]
    eta_vals = np.arange(0.01, 1.0, 0.01)
    # eta_vals = np.delete(eta_vals, 5)  # error here
    
    data_array_mat1 = np.zeros([len(eta_vals), len(r_vec)], dtype=np.float64)
    data_array_mat2 = np.zeros([len(eta_vals), len(r_vec)], dtype=np.float64)
    for i, eta in enumerate(eta_vals):
        outd_eta = f'./output/{run_id}_{int(100*eta):03}eta/'
        fname1 = outd_eta + dect_id + f'/mat1_{int(t_1):02}tiss_{int(t_2):02}bone.bin'
        fname2 = outd_eta + dect_id + f'/mat2_{int(t_1):02}tiss_{int(t_2):02}bone.bin'
        data_array_mat1[i,:] = np.fromfile(fname1, dtype=np.float64)
        data_array_mat2[i,:] = np.fromfile(fname2, dtype=np.float64)
        #%%
    fig, ax = plt.subplots(1,1, figsize=[5,3])
    for i, r in enumerate(r_vals):
        ind_r = int(100*r) - 1  #list(r_vec).index(r)
        data_mat1 = data_array_mat1[:,ind_r] * np.sqrt(eta_vals)
        data_mat2 = data_array_mat2[:,ind_r]  * np.sqrt(eta_vals)
        
        #x.plot(eta_vals, data_mat1, color='b', label=r)
        ax.plot(eta_vals, data_mat2, label=r)
    
    ax.set_ylabel('SNR for given allocation $r$' )
    ax.set_xlabel('detective efficiency $\eta$')
    ax.legend()
    fig.tight_layout()
    if savefig:
        plt.savefig(figd_r1+f'r1_plot2_mvkv_snr_v_eta_{int(t_2):02}bone.pdf')
    plt.show()
        
    
    
    
    
    







        
        