#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 08:42:04 2022

@author: giajadick

This file includes the classes for different elements of the imaging system:
    1. Material    (for any object slab)
    2. Object      (made up of multiple Materials)
    3. Source      (polychromatic x-ray spectrum)
    4. Detector    (a Material with corresponding efficiency functions)
    
The Material class includes methods for getting x-ray mass attenuation and 
mass energy absorption coefficient data is from NIST: 
https://physics.nist.gov/PhysRefData/XrayMassCoef/tab4.html)
Each material file is a .txt with three columns (energy, mu/rho, mu_en/rho). 
This is the same formatting that NIST uses for each material's data.
More materials can be added by creating a new .txt file with the data 
copied in this format. There's probably an easier way to scrape this data 
without manual copying, but it is easy enough for a small number of materials.
    
This file also includes the functions for calculating the signal and variance
detected after a Source passes through an Object to a Detector.
This informs functions for calculating partial derivatives used in setting 
up each Fisher information matrix to get the Cramer-Rao Lower Bound. 
"""

import numpy as np
from scipy.io import loadmat
import pandas as pd

# for mass attenuation coefficients
from input.materials.MAC_interp import get_MAC, get_MAC_en


class Material:
    '''
    Class handling a single slab of known thickness/density/atomic composition.
    '''
    def __init__(self, name, rho, t):
        self.name = name   # material name
        self.rho = rho     # density [g/cm3]
        self.t = t         # thickness [cm]
        self.A = rho*t     # mass thickness [g/cm2]
    
    def mass_atten(self, E):
        '''mu/rho [cm2/g], mass attenuation coefficients evaluated at E'''
        return get_MAC(self.name, E)
    
    def mass_energy(self, E):
        '''mu_en/rho [cm2/g], mass energy absorption coefficients evaluated at E'''
        return get_MAC_en(self.name, E)


class Object: 
    '''
    Class handling an object of many slabs, assumed to be arranged sequentially.
        2. Object, transmission function T
        (E) and transmitted spectrum I(E)
    '''
    def __init__(self, materials):
        '''
        materials : list of Materials
        '''
        self.materials = materials
        
    def get_transmission_function(self, source):
        T = np.exp(np.sum([-mat.A * mat.mass_atten(source.E) for mat in self.materials], axis=0))
        return T
    
    

class Source:
    '''
    Class for handling the spectrum at all three stages of the imaging process:
        1. Source, initial spectrum I0(E)
        
    '''
    def __init__(self, filename, name):
        
        self.filename = filename 
        self.name = name
        
        # attempt reading file
        try:  
            if 'Accuray' in filename:
                spec_data = pd.read_csv(filename, sep=',')
                self.I0_raw = spec_data['Weight'].to_numpy()
                self.E = spec_data['MeV'].to_numpy()*1000 # convert MeV -> keV
            else:
                spec_data = loadmat(filename)
                self.I0_raw = spec_data['ss'][:,0]
                self.E = spec_data['ee'][:,0]
        except:
            print(f"Failed to open spectrum filename {filename}, failed to initialize.")
        
        # normalize counts
        self.I0 = self.I0_raw #/np.trapz(self.I0_raw, x=self.E)
        
    def rescale_I0(self, scale):
        self.I0 = self.I0 * scale

    def get_water_dose(self, water):
        '''
        Get the estimated dose through some water-equivalent attenuation.
            water : (Object)
        '''
        # get spectra attenuation through water
        T_water = np.exp(-water.A * water.mass_atten(self.E))
        I_water = self.I0 * T_water
        
        # get MAC_en for water
        MAC_en = water.mass_energy(self.E)
        
        # get dose
        dose = np.trapz(self.E * I_water * MAC_en, x=self.E)    # assume CPE (Attix Eqn 11.1)
        dose *= 1.602e-13   # convert from keV/g to Gy
        return dose


        
class Detector:
    '''
    3. Detector, detection function D(E) and detected signal lambda

    '''
    def __init__(self, filename, detector_mode, ideal):
        '''
        detector_filename : (str) path to detector efficiency file, float32 array of E, eta(E)
        detector_mode : (str) 'PCD' or 'EID'
        ideal : (bool) whether the detector stops every photon
        '''
        self.mode = detector_mode
        self.ideal = ideal
        self.filename = filename
        
        # read file
        data = np.fromfile(filename, dtype=np.float32)
        self.N_energy = len(data)//2
        self.E = data[:self.N_energy]      # 1st half is energies
        self.eta = data[self.N_energy:]    # 2nd half is detective efficiencies
        if ideal:
            self.eta = np.ones(self.N_energy, dtype=np.float32)  # if ideal, switch to ones

    def get_psi(self, source, alpha=1):
        '''
        get mode-specific weighting function (energy integrating vs counting)
        
            source : (Spectrum)
        '''
        if self.mode == 'PCD':
            psi = np.ones(len(source.E))
        else:
            psi = alpha * source.E
        return psi
        
    def get_detection_function(self, source):
        '''
        source : (Spectrum)
        '''
        # interpolate efficiency to source energies
        eta = np.interp(source.E, self.E, self.eta)
        
        # get mode-specific psi
        psi = self.get_psi(source)
            
        return eta * psi
    
    
    
def get_signal(source, target, detector):
    
    T = target.get_transmission_function(source)
    D = detector.get_detection_function(source)
    y = np.trapz(source.I0 * T * D, x=source.E)

    return y



def get_variance(source, target, detector):
    # EID only 
    if detector.mode != 'EID':
        print("Error, variance calculation is for EID mode only")
        return -1
    
    T = target.get_transmission_function(source)
    D = detector.get_detection_function(source)
    psi = detector.get_psi(source)
        
    # variance has factor of psi**2 (one is already contained in D)
    v = np.trapz(psi * source.I0 * T * D , x=source.E)

    return v



def get_signal_partials(source, target, detector):
    # calculate partipal derivatives of signal wrt each material in target
    # returns a list of all partials (dy/dA)
    # true for both EID and PCD detector modes
    
    T = target.get_transmission_function(source)
    D = detector.get_detection_function(source)
    
    signal_partials = []
    for mat in target.materials:
        # for each partial, pull out extra factor of -MAC
        dy_dA = -np.trapz( mat.mass_atten(source.E) * source.I0 * T * D, x=source.E )
        signal_partials.append(dy_dA)
            
    return signal_partials



def get_variance_partials(source, target, detector):
    T = target.get_transmission_function(source)
    D = detector.get_detection_function(source)
    psi = detector.get_psi(source)
    
    variance_partials = []
    for mat in target.materials:
        # for each partial, pull out extra factor of -MAC * psi
        dv_dA = -np.trapz( psi * mat.mass_atten(source.E) * source.I0 * T * D, x=source.E )
        variance_partials.append(dv_dA)
        
    return variance_partials



def get_CRLBs_PCD(spec_a, spec_b, target, detector):
    '''
    Poisson noise CRLB calculation
    '''
    # assumes two materials
    N_mat = len(target.materials)
    if N_mat > 2:
        print(f"Error, CRLB calculation is for 2 materials (currently using {N_mat}")
        return -1
    
    y_a = get_signal(spec_a, target, detector)
    y_b = get_signal(spec_b, target, detector)
    
    # m_ji = partial derivative of spectrum j w.r.t. material i
    m_11, m_12 = get_signal_partials(spec_a, target, detector)
    m_21, m_22 = get_signal_partials(spec_b, target, detector)
    
    # get CRLB for each material
    CRLB_1 = (y_a*m_22**2 + y_b*m_12**2) / (m_11*m_22 - m_12*m_21)**2
    CRLB_2 = (y_a*m_21**2 + y_b*m_11**2) / (m_11*m_22 - m_12*m_21)**2

    return np.array([CRLB_1, CRLB_2])



def get_CRLBs_EID(spec_a, spec_b, target, detector):
    '''
    Compound Poisson noise CRLB calculations (approx as Gaussian)
    '''
    # assumes two materials
    N_mat = len(target.materials)
    if N_mat > 2:
        print(f"Error, CRLB calculation is for 2 materials (currently using {N_mat}")
        return -1
    
    # get signal variances
    v_a = get_variance(spec_a, target, detector)
    v_b = get_variance(spec_b, target, detector)
        
    # get m_ji (partial derivative of signal j w.r.t. material i)
    m_11, m_12 = get_signal_partials(spec_a, target, detector)
    m_21, m_22 = get_signal_partials(spec_b, target, detector)
    
    # get v_ji (partial derivatives of variance j w.r.t material i)
    v_11, v_12 = get_variance_partials(spec_a, target, detector)
    v_21, v_22 = get_variance_partials(spec_b, target, detector)
    
    # Fisher matrix elements
    F11 = (1/v_a)*m_11*m_11 + (1/v_b)*m_21*m_21 + (1/2)*( (1/v_a**2)*v_11*v_11 + (1/v_b**2)*v_21*v_21 )
    F12 = (1/v_a)*m_11*m_12 + (1/v_b)*m_21*m_22 + (1/2)*( (1/v_a**2)*v_11*v_12 + (1/v_b**2)*v_21*v_22 ) 
    F21 = F12
    F22 = (1/v_a)*m_12*m_12 + (1/v_b)*m_22*m_22 + (1/2)*( (1/v_a**2)*v_12*v_12 + (1/v_b**2)*v_22*v_22 ) 
    
    # get CRLB for each material
    F_det = F11*F22 - F12*F21   # determinant
    CRLB_1 = F22/F_det
    CRLB_2 = F11/F_det
    
    return np.array([CRLB_1, CRLB_2])


   
    
    
    
