# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 16:21:48 2016

@author: php
"""

# a bunch of packages for simple life
import numpy as np
import scipy as sp
# image : PIL et creation image sp
from PIL import Image
from scipy import ndimage
from scipy import signal

from sklearn.metrics import auc

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.gridspec as gridspec
import pandas as pd
import os
import time

# data for processing
inputpath = '/home/as/Bureau/results/force-stamp/relaxation/'

# infos : couleurs voulues
coloriage=['black', 'green', 'blue', 'red']
# infos : tri des données = 1 courbe, c'est une ?
typetri=" cells"


# recuperation liste des repertoires et tri par alphabeta
repertoires = [f for f in os.listdir(inputpath)]
repertoires.sort()
#print repertoires

# preparation de la figure
fig=plt.figure("Plot", figsize=(7,7))

# index pour le choix des couleurs
i=0
# recuperation des listes de fichiers des differents repertoires
for dossier in repertoires:
    localdossier=inputpath+str(dossier)+'/'
    # liste des fichiers au bon format
    files = [f for f in os.listdir(localdossier) if f.endswith('.txt')]
    # tri par ordre alphabetique
    files.sort()
    # creation des dataframes pour les moyennes en force et en temps pour eviter les pbs de longueur de fichier via NaN
    localmean=pd.DataFrame()
    localtime=pd.DataFrame()

    # boucle secondaire sur les fichiers
    for fichier in files:
        localfichier=localdossier+fichier
	#load data
        df = pd.read_csv(localfichier, delimiter=r"\s+",   comment='#', names=['f', 't'], skiprows=72, skipfooter=66)
        force=df['f']*10**12
	#on remet le premier point au meme niveau
        forcecorr=force-force.max()
        choix=pd.DataFrame(forcecorr) #ratio
        localmean=pd.concat([localmean, choix], axis=1)
        localtime=pd.concat([localtime, df['t']], axis=1)
    # on trace les courbes moyennes avec sans SD
    plt.scatter(localtime.mean(axis=1),localmean.mean(axis=1)-localmean.std(axis=1), color=coloriage[i], alpha=0.01, s=1)
    plt.scatter(localtime.mean(axis=1),localmean.mean(axis=1), color=coloriage[i], alpha=0.25, s=2)
    plt.scatter(localtime.mean(axis=1),localmean.mean(axis=1)+localmean.std(axis=1), color=coloriage[i], alpha=0.01, s=1)
    # on rajoute la correspondance condition / couleur / nombre
    plt.text(3, -600-i*25, dossier+" "+str(len(localmean.columns))+ typetri, color=coloriage[i])
    #print dossier + " is in "+coloriage[i]
    i=i+1
#on met tout ca propre avec les axes
plt.ylim(-750, 50)
plt.xlim(-0.5, 5.5)
plt.xlabel("Temps (sec)")
plt.ylabel("Relaxation : F-F(t=0) (pN)")

#elegance
fig.tight_layout() 
#sortie ecran
plt.show()
