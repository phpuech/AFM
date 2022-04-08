# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 22:40:53 2015

@author: php
"""

"""

ne contient pas de correction pour la pente de la ligne de base

"""

#---------------------------------------------------

# a bunch of packages for simple life
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.gridspec as gridspec
import pandas as pd
import os

#---------------------------------------------------

# outlook for plots
#plt.rcParams.update({'font.size': 16})
#plt.rcParams.update({'font.family':'serif'})
caract= 11

#---------------------------------------------------

# where is the file
inputpath = '/home/php/Bureau/'
#fichier = inputpath + 'verre.txt'
fichier = inputpath + 'test.txt'

#---------------------------------------------------
#parametres mesures
nomlevier = "MLCT 10"
# speed (um/s)
vitesse = 70.
# ALL : corrected & used lever spring constant
k = 12.7 #pN/nm

#---------------------------------------------------
#---------------------------------------------------
viscosite = 10**(-3) # N/m**2, eau ou pbs

# fonction de fit Janovjak 2005
initialguess = [1,1]
def Fit(x, *p):
    fit=np.ones(len(x))
    j=0
    for i in x:
        fit[j]=p[0]/(p[1]+i)
        j=j+1
    return fit

#---------------------------------------------------

print "Calculating parameters for HydroDrag, following Janovjak 2005 "  

#---------------------------------------------------
# loading txt containing
# avec h la distance au dessus de la surface (ie TSS)
# h (nm) Fh (pN)
# avec une ligne de commentaires
# 
df = pd.read_csv(fichier, delimiter=r"\s+",   comment='#', names=['h', 'fh'], skiprows=1)

#---------------------------------------------------

# create main figure  
fig1=plt.figure(figsize=(5,5), dpi=100)
ax1 = plt.subplot(111)
#---------------------------------------------------
plt.title("Drag parameters", fontsize=12)

# preparation des data
# en um
x = df['h']*10**(-9)#4.625
# en N
force = df['fh']*10**(-12) # pas de correction ici pour la ligne de base 

# plot corrected for TSS
plt.plot(df['h']*10**(-3), df['fh'], 'o', color='red', alpha=0.5)
# set the plot
plt.ylabel('Fh (pN)', fontsize = caract)
plt.xlabel('Distance (um)', fontsize = caract)
plt.xlim(0,)
plt.ylim(0,1000)

fitParams, fitCovariances = curve_fit(Fit, x, force, p0=initialguess)

aeff= np.round(10**6*np.sqrt(fitParams[0]/(6.*np.pi*viscosite*vitesse*10**(-6))), decimals=3)

beff=np.round(fitParams[1]*10**6, decimals = 3)

print "Coefficients"
print "aeff (um) =", aeff
print "beff (um) =", beff

fitted = Fit(x, fitParams[0], fitParams[1])
    # plot it
plt.plot(df['h']*10**(-3), fitted*10**12, '-', color="black")

texte="Lever type : " + nomlevier + "\n"+"aeff ="+ str(aeff) + " um\n"+"beff = "+str(beff)+" um"
ax1.text(1,800, texte, fontsize=10)

#---------------------------------------------------
fig1.tight_layout()

plt.show()
