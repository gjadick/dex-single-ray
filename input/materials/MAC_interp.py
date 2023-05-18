#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 13:15:07 2022

@author: glj
"""

import os 
import numpy as np

rootpath = os.path.dirname(__file__)
if os.path.exists(rootpath) == False:
    print('unable to find root directory, using:')
    print(rootpath)

def get_material_data(material_name):
    '''
    Parameters
    ----------
    material_name : name of material (str)

    Returns 
    -------
    data: lists of energy [MeV], MAC [cm^2/g], MAC_en [cm^2/g] (np array)

    '''
    material_file = os.path.join(rootpath, material_name+'.txt')

    f = open(material_file, 'r')
    # data in order [MeV, MAC, MAC_en]
    data = np.array([line.split() for line in f.readlines()], 
                    dtype=np.float32).T
    f.close()

    return data

def get_MAC(material_name, E):
    '''

    Parameters
    ----------
    material_name : name of material (str)
    E : list of energies [keV]

    Returns
    -------
    linearly interpolated MAC values at E

    '''
    MeV, MAC, MAC_en = get_material_data(material_name)
    keV = MeV*1000
    return np.interp(E, keV, MAC)


def get_MAC_en(material_name, E):
    '''

    Parameters
    ----------
    material_name : name of material (str)
    E : list of energies [keV]

    Returns
    -------
    linearly interpolated MAC_en values at E

    '''

    MeV, MAC, MAC_en = get_material_data(material_name)
    keV = MeV*1000
    return np.interp(E, keV, MAC_en)
    
    
    
    
    
    
    
    