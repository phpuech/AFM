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
inputpath = '/home/php/Bureau/FCs/'
#fichier = inputpath + 'verre.txt'
fichier = inputpath + 'fast.txt'

#---------------------------------------------------
#parametres mesures

# speed (um/s)
vitesse = 10
# ALL : corrected & used lever spring constant
k = 12.7 #pN/nm

#---------------------------------------------------
# model name : hdependance, simplified
model = "simplified" # ATTENTION MARCHE PAS BIEN CF. PLUS BAS EQUATION BOF
#model = "complete"
#---------------------------------------------------
viscosite = 10**(-3) # N/m**2, eau ou pbs
# mise en place du facteur de Fh= (facteur) * v
# measured parametersfor large triangular lever (MLCT)
aeff=52.06 # (um) -- Janovjak 2005
deff=5.48 # (um) -- Janovjak 2005
#globalfacteur=2.0 #(pN/(um/s)) -- Moy papers, Zang 2002
globalfacteur=8.1 #(pN/(um/s)) -- PHP / AS mesures 2015 ou Janovjak 2005

if model == "simplified":
    def facteur(x):
        return  globalfacteur
elif model == "complete":
    def facteur(x): 
        #attention x doit etre en um au vu des dimensions de aeff et deff
        taille = len(x)
        facteurcomplet = np.ones(taille)
        i=0
        while i<taille:
            facteurcomplet[i]=(6*np.pi*viscosite*aeff**2)/(x[i]+deff)
            i=i+1
        return facteurcomplet
    

#facteur=8.7

#---------------------------------------------------

print "Correcting for HydroDrag, version ", model    

#---------------------------------------------------
# loading txt export RETRACE ONLY files with all channels
# h is the measured smoothed height
# h2 is the measured height
# h3 is the height : DO NOT USE
# t is segment time
# t2 is series time
# from exported file
# fancyNames: "Height (measured & smoothed)" "Vertical Deflection" "Height (measured)" "Height" "Lateral Deflection" "Series Time" "Segment Time"
df1 = pd.read_csv(fichier, delimiter=r"\s+",   comment='#', names=['h', 'f','h2', 'h3', 't', 't2'], skiprows=74)

#---------------------------------------------------

# create main figure  
fig1=plt.figure(figsize=(14, 7), dpi=100)
#---------------------------------------------------
plt.subplot(231)
plt.grid()
plt.title('Original data, non smoothed')

# preparation des data
# Tip sample separation in um
TSS = (df1['h']*10**6 + (df1['f']*10**12 / (1000*k)))
#print TSS.values
# corrected for contact point BY HAND
#x= TSS-(TSS[0]-3.9863)
x = TSS-TSS[0]#4.625
#non corrected deflection in pN
# baseline correction et/ou retournement de la courbe 
force = -df1['f']*10**12 # pas de correction ici pour la ligne de base 

# plot corrected for TSS
plt.plot(x, force, '-', color='red', alpha=0.5)
# set the plot
plt.ylabel('Force (pN))', fontsize = caract)
plt.xlabel('Z-d (um)', fontsize = caract)
#plt.xlim(0,1.5)
#plt.ylim(,)

#---------------------------------------------------

# TSS vs time  
#fig2=plt.figure('Speed calculation', figsize=(7, 7), dpi=100)
plt.subplot(232)
plt.grid()
plt.title('Tip displacement')
temps = df1["t"]-df1["t"][0]
# plot corrected for TSS
plt.plot(temps,x,  '-', color='red', alpha=0.5)
# set the plot
plt.ylabel('Z-d (um))', fontsize = caract)
plt.xlabel('t (sec)', fontsize = caract)
plt.xlim(0,)
plt.ylim(0,)

#---------------------------------------------------
# calculation of the derivative
# approximation
# avec la methode diff de pandas, de proche en proche
# attention le recalage des datas est incertain... NaN trainent
speed = x.diff(periods=1) / temps.diff(periods=1)
#print speed
#---------------------------------------------------
# speed vs time  
#fig3=plt.figure('Derivative vs time', figsize=(7, 7), dpi=100)
plt.subplot(233)
plt.grid()
plt.title('Tip speed')
temps = df1["t"]-df1["t"][0]
# plot corrected for TSS
plt.plot(temps,speed,  '-', color='red', alpha=0.5)
# set the plot
plt.ylabel('Tip speed = d(Z-d)/dt (um/s)', fontsize = caract)
plt.xlabel('t (sec)', fontsize = caract)
plt.xlim(0,)
#plt.ylim(0,1)
# on debruite un peu a cause du bruit intrinseque
fenetre = 10
corry=pd.rolling_mean(speed, fenetre)    
corrx=pd.rolling_mean(temps, fenetre)
plt.plot(corrx,corry,  '-', color='red', alpha=0.5)

#---------------------------------------------------
# calcul de la force, d'apres Janovjak 2005
tip  = corry * facteur(TSS)
levier = np.ones(len(corry))*vitesse * facteur(TSS)
if model == "complete":
    deltaf=levier -tip
elif model == "simplified":
    deltaf=facteur(TSS)*(vitesse-speed)
minimaldeltaf=deltaf.min()
#print "Maximal correction (pN) = ", minimaldeltaf

#---------------------------------------------------
# plot des corrections
#fig4=plt.figure('force vs. TSS pour correction', figsize=(7, 7), dpi=100)
plt.subplot(234)
plt.grid()
plt.title('Correction forces')

plt.plot(temps,tip,  '-', color='red', alpha=0.5, label = "tip")
plt.plot(temps,levier,  '-', color='black', alpha=0.5, label = "lever")
plt.legend(loc='lower right', frameon=False)

plt.xlabel('Z-d (um)', fontsize = caract)
plt.ylabel('Force (pN)', fontsize = caract)
plt.xlim(0,)

#fig5=plt.figure('Delta Force vs. TSS pour correction', figsize=(7, 7), dpi=100)
plt.subplot(235)
plt.grid()
plt.title('Delta F (lever-tip)')
plt.plot(temps,deltaf,  '-', color='red', alpha=0.5)
#corrforce=corrx=pd.rolling_mean(deltaf, fenetre)
#plt.plot(temps,corrforce,  '-', color='black', alpha=0.5)

plt.xlabel('Z-d (um)', fontsize = caract)
plt.ylabel('DeltaForce (pN)', fontsize = caract)
plt.xlim(0,)

# ATTENTION : COMMENT EST CE QUE L'AXE X EST GERE ??? ALIGNEMENT ???
# SI SAUT DE FORCE SUR DEUX POINTS ALORS PAS PLUS QUE PENTE SUR DEUX POINTS
#fig6=plt.figure('Final force', figsize=(7, 7), dpi=100)
plt.subplot(236)
plt.grid()
plt.title('Data corrected for HydroDrag')
plt.plot(temps,force,  '-', color='red', alpha=0.5, label='uncorrected')
plt.plot(temps,force + deltaf,  '-', color='black', alpha=0.5, label='corrected')
plt.legend(loc='lower right', frameon=False)
plt.xlabel('Z-d (um)', fontsize = caract)
plt.ylabel('Force (pN)', fontsize = caract)
plt.xlim(0,)
#

# esthetical for the entire plot
fig1.tight_layout()
#fig2.tight_layout()
#fig3.tight_layout()
#fig4.tight_layout()
#fig5.tight_layout()
#fig6.tight_layout()

plt.show()
