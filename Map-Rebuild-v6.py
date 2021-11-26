# -*- coding: utf-8 -*-
"""
From MJD code v4
Modified PHP march 2019
converted with 2to3

@author: php
"""

# test pour reconstruction map de mecanique

#---------------------------------------------------

# a bunch of packages for simple life
import numpy as np
import scipy as sp
# image : PIL et creation image sp
from PIL import Image
from scipy import ndimage

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.gridspec as gridspec
import pandas as pd
import os
import time
import seaborn as sns
#from skimage import data, io, filters

#---------------------------------------------------

#on recupere les dates et heures de compilation
now = time.strftime("%c")
today =time.strftime("%Y")+time.strftime("%m")+time.strftime("%d")
heure = time.strftime("%H")+time.strftime("%M")+time.strftime("%S")
maintenant = today + "-" + heure
print(now)
print("---------------------------------------------------")
#---------------------------------------------------

# outlook for plots
plt.rcParams.update({'font.size': 12})
#plt.rcParams.update({'font.family':'serif'})

#---------------------------------------------------


def couper(l, n):
    # decoupage d'une liste en sous listes de meme taille AVEC INVERSION DU SENS UNE FOIS SUR DEUX
    temp=0
    for i in range(0, len(l), n):
        if temp%2==0:
            yield l[i:i+n]
            #print(l[i:i+n])
            temp=temp+1
        else:
            var = l[i:i+n]
            rev = var[::-1]
            yield rev
            #print(rev)
            temp=temp+1
        

#-----------------------------------------
# TO BE MODIFIED BY USER
# where is the file
folder = '/home/php/Bureau/test/adhesion/20190308-155919-adhesion-results/'
nom='20190308-155919-data.txt'
experiment='adhesion' # adhesion, mechanics
condition="F127 vs. F127" # F127 vs. F127, 500Pa
# taille d'une ligne en pixels pour une image carree
line=8
colonne=8
# taille de l'image en microns pour une image carree
size=48
#extreme values
yminmap=0
ymaxmap=500
yminhist=0
ymaxhist=500
#-----------------------------------------

#-----------------------------------------
inputpath=folder+nom
if experiment=='mechanics':
    titre="Nominal="+condition
else:
    titre = "Treatment="+condition

if experiment=='mechanics':
    cartedata = pd.read_csv(inputpath, delimiter=r"\s+",   comment='#', names=['nom', 'E','dE', 'contact','dcontact', 'rms', 'curve'], skiprows=9)
else:
    cartedata = pd.read_csv(inputpath, delimiter=r"\s+",   comment='#', names=['nom', 'E'])
# test=np.arange(line*line) # create an artificial map to check the classification of the pixels

#----------------------------------------------------------------------------------

imageMap=plt.figure('Map', dpi=150)

# interlude pedagogique
#solution intuitive
#data=np.array(carte['E']).reshape(-1, line)
# ATTENTION CELA NE MARCHE PAS : l'afm ecrit les points en S... et non pas de la gauche veres la droite a chaque fois
# la bonne solution est d'utiliser la fonction creer qui trie les lignes paires et impaires 
data=list(couper(cartedata['E'].values, line))

ax=sns.heatmap(data, cmap='viridis', square=True, vmin=yminmap, vmax=ymaxmap, xticklabels=False, yticklabels=False, cbar=False)
#retourner l'image pour avoir (0,0) en bas a gauche
ax.invert_yaxis()

# changement des valeurs de facon a avoir les tailles dans le bon sens
avant=[0,line]
apres=[0, size]
plt.xticks(avant, apres)
plt.yticks(avant, apres)
plt.xlabel('x (µm)')
plt.ylabel('y (µm)')
plt.title(titre)
# modification de la barre de couleurs, avec legende
cbar = ax.figure.colorbar(ax.collections[0])
if experiment=='mechanics':
    cbar.set_label('Young modulus (Pa)')#, rotation=270)
else:
    cbar.set_label('Adhesion force (pN)')#, rotation=270)

#----------------------------------------------------------------------------------

imageHist=plt.figure('Hist', figsize=(7,4), dpi=150)
sns.distplot(cartedata['E'])
plt.axvline(x=np.round(cartedata['E'].median(), 1), color='red')
if experiment=='mechanics':
    plt.xlabel('Measured E (Pa)')
else:
    plt.xlabel('Adhesion force (pN)')
#plt.xlim(0,)
#plt.xscale("log")
plt.xlim(yminhist,ymaxhist)
plt.ylim(0,0.02)
plt.ylabel('Prob. density')
plt.title(titre)
texte="median="+str(np.round(cartedata['E'].median(), 1))+"Pa"
plt.text(25,0.01, texte, fontsize=12, color='red') 
print("Median value", np.round(cartedata['E'].median()))
# esthetique
imageMap.tight_layout()
imageHist.tight_layout()

plt.show()
        
