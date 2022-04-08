# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 13:44:24 2015

Corrected PHP June 2016 for d0

@author: php
"""

"""

ne contient pas de correction de ligne de base

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
import time
#from statsmodels import eval_measures

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
#methode de traitement des donnees
methodpath = '/home/php/Bureau/test2/'
methofilename = 'method.txt'
methodloc= methodpath+methofilename
# data for processing
inputpath = '/home/php/Bureau/test2/fc/'
outputpath = inputpath + maintenant + '-results/'
datasave = outputpath+maintenant +'-data.txt'
#on peut aussi choisir de sauver ailleurs...

#---------------------------------------------------

methode = open(methodloc, 'r')
methodelines = methode.read().splitlines() # avoid the \n at the end of the file
#print methodelines
methode.close()
# model name : pyramid, sphere, linear (for tension), cone
# LACKS : CYLINDER
model = methodelines[1]
# formatet or not  du fichier : subset or all
fichiertype = methodelines[3]
# force initiale (pN) in order to correct for wrong value due to baseline shift : fcorr=f+initiaforce-f0
initialforce = float(methodelines[5])
# t0 pour fit si utilisé cf Husson Sci Rep 2016
t0 = float(methodelines[7]) #pN/nm
# number of lines to process for fit
lignes = float(methodelines[9])

#---------------------------------------------------

if not os.path.exists(outputpath):
    os.makedirs(outputpath)
# fichier de sortie des datas
file = open(datasave, "w")
file.write("# " + now+"\n")
file.write("#---------------------------------------------------\n")
# liste des fichiers au bon format
files = [f for f in os.listdir(inputpath) if f.endswith('.txt')]
# tri par ordre alphabetique
files.sort()

# output commande line
print "Fitting with", model, "model"
print "---------------------------------------------------"
# to output file
file.write("#---------------------------------------------------\n")
file.write("# Fitting with "+ model+ " model\n")
file.write("# over "+ str(lignes)+ " data points\n")
file.write("# Fmax = "+ str(initialforce)+ " model\n")
file.write("#---------------------------------------------------\n")
# preparation du fichier de sortie, separe par des espaces
if model == "power":
    prefacteur="A (N)"
else:
    prefacteur="log (A/Fmax)"
file.write("# Filename | Prefactor | alpha | t0 (sec)\n")
file.write("#---------------------------------------------------\n")

#---------------------------------------------------

# model definition
if model == "linear":
    initialguess = [2,-0.1]
    def Fit(x, *p):
        fit=np.ones(len(x))
        j=0
        for i in x:
            fit[j]=p[0]+i*p[1]
            j=j+1
        return fit
    xlab="log(t/t0)"
    ylab="log(F/Fmax)"
elif model=="power":
    initialguess = [2,1, -0.1]
    def Fit(x, *p):
        fit=np.ones(len(x))
        j=0
        for i in x:
            fit[j]=p[0]*(i/p[1])**p[2]
            j=j+1
        return fit
    xlab="t/t0"
    ylab="F (N)"


#---------------------------------------------------
# loading txt export relaxation only with all channels
# h is the measured smoothed height
# h2 is the measured height
# h3 is the height : DO NOT USE
# from JPK exported file
# fancyNames: "Height (measured & smoothed)" "Vertical Deflection" "Height (measured)" "Height" "Lateral Deflection" "Series Time" "Segment Time"
for fichier in files:
    localfichier=inputpath+fichier
    if fichiertype == "all":
        df1 = pd.read_csv(localfichier, delimiter=r"\s+",   comment='#', names=['h', 'f','h2','h3', 't', 't2'], skiprows=74, nrows=lignes)
    else:
        df1 = pd.read_csv(localfichier, delimiter=r"\s+",   comment='#', names=['f','t2'], skiprows=74, nrows=lignes)

    #print(df1.head())
    #---------------------------------------------------
    # definition of plot as a grid to have the residues above the fitted curve
    gs = gridspec.GridSpec(2, 1,height_ratios=[1,3])
    # for axis label localization, to align the labels
    labelx = -0.1
    #---------------------------------------------------

    # create main figure  
    fig1=plt.figure(fichier, figsize=(7, 7), dpi=100)
    #---------------------------------------------------

    # first sub
    ax1 = plt.subplot(gs[1])
    #---------------------------------------------------
    # data preparation before plot and fit

    # conversion to usual values
    x = df1['t2'] # piezo en  microns
    y = df1['f']#-df1['f'].min()# des fois que ca soit pas la bone valeur ou qu'on ait oublie de mettre la baseline a zero

    # logs avec meme nom de variable pour eviter de tout reprendre dans la partie plot
    if model == "linear":
        temps=np.log10(x/t0) 
        piconewtons=np.log10((y + (initialforce*10**-12) - y[0])/(initialforce*10**-12))
    else:
        temps=x
        piconewtons=y + (initialforce*10**-12) - y[0]

    #print piconewtons.head()

    #TSS calculation with no correction on original data
    #converted to nm
    #carefull with sign (depends on global orientation of FC): check with force curve on glass
    # NOTE : TSS et deflection en nm pour FIT

    #---------------------------------------------------

    # plot corrected for TSS
    plt.plot(temps, piconewtons, '-', color='red', alpha=0.5)
    # set the plot
    plt.ylabel(ylab, fontsize = 16)
    plt.xlabel(xlab, fontsize = 16)
    #plt.xlim(0,)
    if model=="power":
        plt.ylim(0,1000e-12)

    # prepare y label alignment
    ax1.yaxis.set_label_coords(labelx, 0.5)


    #---------------------------------------------------
    # FITTING : leastsquare fit to the model
    #with the initial guesses that are in the definition of the fit function
    fitParams, fitCovariances = curve_fit(Fit, temps, piconewtons, p0=initialguess)
    # to see the output of the fit, uncomment the two next lines
    #print "Fit coefficients:", fitParams #\n a*t**2+b*t+c\n', fitParams
    #print "Covariance matrix:\n", fitCovariances

    #---------------------------------------------------
    #recalculate data to superimpose the fit on the original curve
    if model == "linear":
        fitted = Fit(temps, fitParams[0], fitParams[1])
    else:
        fitted = Fit(temps, fitParams[0], fitParams[1], fitParams[2])
    #output data to the console
    print fichier
    if model == "linear":
        print "log(A/Fmax) = ", fitParams[0]
        print "t0 sets to 1"
        print "alpha = ", fitParams[1]
        print "---------------------------------------------------"
    else:
        print "A = ", fitParams[0]
        print "t0 = ", fitParams[1]
        print "alpha = ", fitParams[2]
        print "---------------------------------------------------"

    if model == "power":
        ajout="alpha = "+str(fitParams[2])
        plt.text(0.1,9e-10, ajout)

    # output to file
    # il y a deux cas pour les parametres
    if model == "power":
        t0=fitParams[1]
        prefac=fitParams[0]
        alpha=fitParams[2]
    else:
        prefac=fitParams[0]
        alpha=fitParams[1]
    # output to file
    file.write(fichier+" "+str(prefac)+" "+str(alpha)+" "+str(t0)+"\n")


    # plot it
    plt.plot(temps, fitted, color="black")

    #---------------------------------------------------
    # calculate the error on the fitted parameters (from web)
    error = np.sqrt(np.diag(fitCovariances))
    #---------------------------------------------------

    # second subplot : residues
    ax2 = plt.subplot(gs[0], sharex=ax1)
    # calculate residues
    residus = piconewtons - fitted
    # plot residues of fits
    plt.plot(temps, residus, "-", color="black", alpha=0.5)
    # set the plot
    plt.ylabel('Res.', fontsize = 16)
    #plt.xlim(0,)
    #plt.ylim(-1,1)
    # align y axis label
    ax2.yaxis.set_label_coords(labelx, 0.5)
    # remove ticks of x axis
    plt.setp( ax2.get_xticklabels(), visible=False)
    #---------------------------------------------------

    # esthetical for the entire plot
    fig1.tight_layout() 

    # show graphs on screen
    plotname=outputpath+fichier+'.png'
    plt.savefig(plotname)
    #release memory
    plt.close(fig1)

file.close()