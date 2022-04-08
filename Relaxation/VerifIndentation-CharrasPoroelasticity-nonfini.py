# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 13:44:24 2015

Corrected PHP June 2016 for d0

@author: php
"""

"""

tracé de l'indentation au cours du temps
attention il faut avoir remis la ligne de base a zero

"""


#---------------------------------------------------

# a bunch of packages for simple life
import numpy as np
import scipy as sp
import math as m
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.gridspec as gridspec
import pandas as pd
import os
#from statsmodels import eval_measures


#---------------------------------------------------

# outlook for plots
plt.rcParams.update({'font.size': 16})
#plt.rcParams.update({'font.family':'serif'})

#---------------------------------------------------

# where is the file
#repertoire
inputpath = '/home/php/Bureau/PHP-DEV/Fits/test-TSS/'
#fichier
suffix = '.txt'
nom = 'cal' 
# for the macro, where to find it
fichier = inputpath + nom + suffix

#---------------------------------------------------
# model name : pyramid, sphere, linear, cone
# LACKS : CYLINDER, linear tension
model = "tranquille"
# ALL : corrected & used lever spring constant
k = 11. #pN/nm
# PYR / CONE : half angle (to face for pyr) in degrees
alpha = 15.
# SPHERE / LIN : radius of bead in um
Rb = 5.
# LIN : radius of cell in um
Rc = 5.
#---------------------------------------------------
correction="no"
# compressibilty : non compressible is 0.5
eta = 0.5
#---------------------------------------------------

df1 = pd.read_csv(fichier, delimiter=r"\s+",   comment='#', names=['h', 'f','h2', 'err','h3','h4','l', 't', 't2'], skiprows=74)

fig1=plt.figure(nom, figsize=(7, 7), dpi=100)

microns = df1['h']*10**6 # piezo en  microns
piconewtons = df1['f']*10**12# force en pN
temps=df['t2']

if correction == "yes":
    # creating subset for correction
    # arrondi nombre de points
    zone = np.floor(len(piconewtons)*fraction)
    subsetLin=np.ones(zone)
    a=0
    for i in piconewtons:
        if a < zone : 
            subsetLin[a]=piconewtons[a]
        a=a+1
    LinParams, LinCovariances = curve_fit(LinCorr, microns, subsetLin, p0=initialLin)
    #print LinParams
    piconewtons = piconewtons - (LinParams[0]+LinParams[1]*microns)

#TSS calculation with linear correction on original data if done (felix 06.2016)
#converted to nm
#carefull with sign (depends on global orientation of FC): and on where it happens in the time frame
TSS = 1000*microns + piconewtons / k # k est en pN/nm
#non corrected deflection, in nm
#no baseline correction, no tilt correction for the moment
#deflection = piconewtons / k
#IMPORTANT : to have only positive values, shift the min value to 0, for FIT
#deflection = deflection - np.min(deflection)
#piconewtons2 = piconewtons - np.min(piconewtons)    



# reorient the direction of the force curve (for jpk microscopes)
# this could be helpful for the contact point position
x=TSS[0]-TSS

#---------------------------------------------------

# plot corrected for TSS
#plt.plot(x, piconewtons2, '-', color='blue', alpha=0.25) #non corrected for liln
plt.plot(temps, x, '-', color='red', alpha=0.5) #corr for lin
# set the plot
plt.xlabel('t (sec)', fontsize = 16)
plt.ylabel('Z-d (nm)', fontsize = 16)

fig1.tight_layout() 