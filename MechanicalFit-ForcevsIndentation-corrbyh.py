# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 13:44:24 2015

@author: php
"""
#---------------------------------------------------

from __future__ import division

# a bunch of packages for simple life
import numpy as np
import scipy as sp
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
inputpath = '/home/php/Bureau/test/'
#fichier
suffix = '.txt'
nom = 'cell' 
# for the macro, where to find it
fichier = inputpath + nom + suffix

# hauteur mesuree au point donne en nm
h = 200.

#---------------------------------------------------
# model name : cone corrected for height
model = "cone / corr-h"
# ALL : corrected & used lever spring constant
k = 12.7 #pN/nm
# PYR / CONE : half angle (to face for pyr) in degrees
alpha = 15.
#-------------------------------------------

# compressibilty : non compressible is 0.5
eta = 0.5

#---------------------------------------------------
# model definition

initialguess = [10**2,10**3,1]
def Fit(x, *p):
    fit=np.ones(len(x))
    j=0
    for i in x:
        if i >= p[1]:
            fit[j]=p[2]+p[0]*(i-p[1]+p[2])**2
        else:
            fit[j]=p[2]
        j=j+1
    return fit

    
#---------------------------------------------------
# young modulus calculation
# Sneddon, 1965
def ValYoung(p):
    young = np.round(p*(10**6)*(1.-eta**2)*np.pi/(2.*np.tan(alpha*np.pi/180.)), decimals=1)
    return young
#---------------------------------------------------
# parameter name
if model == "linear":
    parametre = "Tension (microN/m)"
else:
    parametre = "Young modulus (Pa)"
#---------------------------------------------------

# print out the model and parameters
print "Fitting with", model, "model"
print "Half angle =", alpha, ' deg'
print "---------------------------------------------------"
#---------------------------------------------------
# loading txt export TRACE ONLY files with all channels
# h is the measured smoothed height
# h2 is the measured height
# h3 is the height : DO NOT USE
# from JPK exported file
# fancyNames: "Height (measured & smoothed)" "Vertical Deflection" "Height (measured)" "Height" "Lateral Deflection" "Series Time" "Segment Time"
df1 = pd.read_csv(fichier, delimiter=r"\s+",   comment='#', names=['h', 'f','h2', 'h3','l', 't', 't2'], skiprows=74)

#---------------------------------------------------
# definition of plot as a grid to have the residues above the fitted curve
gs = gridspec.GridSpec(2, 1,height_ratios=[1,3])
# for axis label localization, to align the labels
labelx = -0.1
#---------------------------------------------------

# create main figure  
fig1=plt.figure(nom, figsize=(7, 7), dpi=100)
#---------------------------------------------------

# first sub
ax1 = plt.subplot(gs[1])
#---------------------------------------------------
# data preparation before plot and fit

# conversion to usual values
microns = df1['h']*10**6 # piezo en  microns
piconewtons = df1['f']*10**12# force en pN
#TSS calculation with no correction on original data
#converted to nm
#carefull with sign (depends on global orientation of FC): check with force curve on glass
# NOTE : TSS et deflection en nm pour FIT
TSS = 1000*microns + piconewtons / k # k est en pN/nm
#non corrected deflection, in nm
#no baseline correction, no tilt correction for the moment
deflection = piconewtons / k
#IMPORTANT : to have only positive values, shift the min value to 0, for FIT
deflection = deflection - np.min(deflection)
piconewtons = piconewtons - np.min(piconewtons)
# reorient the direction of the force curve (for jpk microscopes)
# this could be helpful for the contact point position
x=TSS[0]-TSS

#---------------------------------------------------

# plot corrected for TSS
plt.plot(x, piconewtons, '-', color='red', alpha=0.5)
# set the plot
plt.ylabel('F (pN)', fontsize = 16)
plt.xlabel('Z-d (nm)', fontsize = 16)
plt.xlim(-50,)
plt.ylim(-5,)
# prepare y label alignment
ax1.yaxis.set_label_coords(labelx, 0.5)


#---------------------------------------------------
# FITTING : leastsquare fit to the model
#with the initial guesses that are in the definition of the fit function
fitParams, fitCovariances = curve_fit(Fit, x, piconewtons, p0=initialguess)
# to see the output of the fit, uncomment the two next lines
#print "Fit coefficients:", fitParams #\n a*t**2+b*t+c\n', fitParams
#print "Covariance matrix:\n", fitCovariances

#---------------------------------------------------
#recalculate data to superimpose the fit on the original curve
fitted = Fit(x, fitParams[0], fitParams[1], fitParams[2])
# plot it
plt.plot(x, fitted, color="black")

#---------------------------------------------------
# calculate the error on the fitted parameters (from web)
error = np.sqrt(np.diag(fitCovariances))
#---------------------------------------------------
# calculate Young modulus and its error
# on a le droit de directement transformer l'erreur car c'est lineaire 
# entre facteur estime, p[0] du fit, et facteur voulu, E ou Tc.
# output to command line

Young = ValYoung(fitParams[0])
errorYoung = ValYoung(error[0])
print "Young modulus (Pa) = ", Young, " +/-", errorYoung

# determine the contact point and its error
contact = np.round(fitParams[1], decimals=1)
errorcontact = np.round(error[1], decimals=1)
#---------------------------------------------------
# output to command line
print "Beginning of non contact zone is at zero displacement" #cf reorientation of FC above
print "Contact point (nm) = ", contact, ' +/-', errorcontact
#---------------------------------------------------

# second subplot : residues
ax2 = plt.subplot(gs[0], sharex=ax1)
# calculate residues
residus = piconewtons - fitted
# plot residues of fits
plt.plot(x, residus, "-", color="black", alpha=0.5)
# set the plot
plt.ylabel('Res. (pN)', fontsize = 16)
plt.xlim(-50,)
plt.ylim(-25,25)
# align y axis label
ax2.yaxis.set_label_coords(labelx, 0.5)
# remove ticks of x axis
plt.setp( ax2.get_xticklabels(), visible=False)
#---------------------------------------------------

# esthetical for the entire plot
fig1.tight_layout() 

# evaluation de la qualite du fit par RMS (cf JPK DP)
N=len(fitted)
carreresidus = residus**2
rms=np.round(np.sqrt(carreresidus.sum()/N), decimals=1)
print "RMS residuals (pN) = ", rms

## plotting the force fitted
#fig2=plt.figure('force test', figsize=(7, 7), dpi=100)
#ax1 = plt.subplot(gs[1])
#plt.plot(x, deflection*k, '-', color='blue', alpha=0.5)
##axes
#plt.ylabel('F (pN)', fontsize = 16)
#plt.xlabel('Z-d (nm)', fontsize = 16)
#ax1.yaxis.set_label_coords(labelx, 0.5)
#plt.plot(x, fitted*k, color="black")
#ax2 = plt.subplot(gs[0], sharex=ax1)
#plt.setp( ax2.get_xticklabels(), visible=False)
#plt.ylabel('Res. (pN)', fontsize = 16)
#plt.xlim(-50,)
#plt.ylim(-25,25)
#ax2.yaxis.set_label_coords(labelx, 0.5)
#plt.plot(x, residy*k, "-", color="black", alpha=0.5)
#fig2.tight_layout() 


# elegance

texte=nom+"\n"+"-------------\n"+"model : "+model+"\n"+"-------------\n"+ parametre +"\n"+ str(Young) + " +/- "+ str(errorYoung) + "\n" + "contact point (nm)\n" + str(contact) + ' +/- ' + str(errorcontact) + "\n" +"RMS residues (pN)\n"+str(rms)
ax1.text(1000,100, texte, fontsize=10)  

# show graphs on screen
plt.show()
