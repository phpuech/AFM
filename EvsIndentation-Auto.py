# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 22:20:32 2015

@author: php
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
import time
#'import seaborn as sns

#---------------------------------------------------

#on recupere les dates et heures de compilation
now = time.strftime("%c")
today =time.strftime("%Y")+time.strftime("%m")+time.strftime("%d")
heure = time.strftime("%H")+time.strftime("%M")+time.strftime("%S")
maintenant = today + "-" + heure
print now
print "---------------------------------------------------"
#---------------------------------------------------

# outlook for plots
plt.rcParams.update({'font.size': 16})
#plt.rcParams.update({'font.family':'serif'})

#---------------------------------------------------

# where is the file
# data for processing
inputpath = '/home/php/Bureau/FCs/mesures/results/'
#datasave = inputpath+maintenant +'-result.txt'
#on peut aussi choisir de sauver ailleurs...

#---------------------------------------------------
#fichiers recuperation et creation de repertoire si necessaire
# fichier de sortie des datas
#file = open(datasave, "w")
#file.write("# " + now+"\n")
#file.write("#---------------------------------------------------\n")
# liste des fichiers au bon format
files = [f for f in os.listdir(inputpath) if f.endswith('.txt')]
# tri par ordre alphabetique
files.sort()

fig1=plt.figure("Compilation", figsize=(12, 6), dpi=100)
ax1 = plt.subplot(121)
df=pd.DataFrame()# creation vide
for fichier in files:
    localfichier=inputpath+fichier
    #---------------------------------------------------
    # loading txt export TRACE ONLY files with all channels
    # h is the measured smoothed height
    # h2 is the measured height
    # h3 is the height : DO NOT USE
    # from JPK exported file
    # fancyNames: "Height (measured & smoothed)" "Vertical Deflection" "Height (measured)" "Height" "Lateral Deflection" "Series Time" "Segment Time"
    df1 = pd.read_csv(localfichier, delimiter=r"\s+",   comment='#', names=['d', 'E','error'], skiprows=1)
    plt.errorbar(df1['d'],df1['E'], yerr= [df1['error'], df1['error']], fmt='-o', alpha=0.5)#, color='black', alpha=0.5)
    df=df.append(df1)# il faut stocker sinon ca marche pas
plt.xlabel('Indentation (nm)', fontsize = 16)
plt.ylim(0,)
plt.xlim(0,)
plt.ylabel('Young modulus (Pa)', fontsize = 16)
plt.title('Separated curves', fontsize = 16)

#fig2=plt.figure("Mean", figsize=(7, 7), dpi=100)
ax2 = plt.subplot(122)
#print df
#plt.plot(df['d'], df['E'], 'o', alpha=0.25, color='black')
grouped=df.groupby(df['d'])
sortedmean= grouped.agg(np.mean)
sortedsd= grouped.agg(np.std)
sortedsd=sortedsd.fillna(0)# sijamais des NaN car une seule data
#print sortedsd
plt.errorbar(sortedmean['d'],sortedmean['E'], yerr= [sortedsd['E'], sortedsd['E']], fmt='-o', alpha=0.5, color='red')
plt.xlabel('Indentation (nm)', fontsize = 16)
plt.ylim(0,)
plt.xlim(0,)
plt.ylabel('Young modulus (Pa)', fontsize = 16)
plt.title('Mean of the group', fontsize = 16)

fig1.tight_layout() 
plt.show()