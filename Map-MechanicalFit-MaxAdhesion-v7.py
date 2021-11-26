# -*- coding: utf-8 -*-
"""
From MJD code v4
Modified PHP march 2019
converted with 2to3

@author: php
"""

# CONTAINS CURVE LENGTH
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
plt.rcParams.update({'font.size': 16})
#plt.rcParams.update({'font.family':'serif'})

#---------------------------------------------------

# where is the file
#methode de traitement des donnees
methodpath = '/home/php/Bureau/Yasab/Code/'
methofilename = 'method-ma.txt'
methodloc= methodpath+methofilename
# data for processing.
# Be sure that the date is on the 'extend' segment.
inputpath ='/home/php/Bureau/test/adhesion/'

#on peut aussi choisir de sauver ailleurs...

# pour le zoom sur les courbes, voici les minima
Xplot=0 # 0 par defaut
#Yplot=-1000
Yres=50

#smooth param for runing mean (odd)
smoothparam=11

#---------------------------------------------------

methode = open(methodloc, 'r')
methodelines = methode.read().splitlines() # avoid the \n at the end of the file
#print methodelines
methode.close()
# quantification type
quantification=methodelines[1]
# model name : pyramid, sphere, linear (for tension), cone
# LACKS : CYLINDER
model = methodelines[3]
# CORRECTION baseline : yes, no ?
correction = methodelines[5]
# on what fraction ?
fraction = float(methodelines[7])
# ALL : corrected & used lever spring constant
k = float(methodelines[9]) #pN/nm
# PYR / CONE : half angle (to face for pyr) in degrees
alpha = float(methodelines[11])
# SPHERE / LIN : radius of bead in um
Rb = float(methodelines[13])
# LIN : radius of cell in um
Rc = float(methodelines[15])
#---------------------------------------------------

# compressibilty : non compressible is 0.5
eta = 0.5

outputpath = inputpath + maintenant + '-'+quantification+'-results/'
datasave = outputpath+maintenant +'-data.txt'

#---------------------------------------------------
#fichiers recuperation et creation de repertoire si necessaire
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

#---------------------------------------------------
# linear correction of baseline on x% of FC

if correction == "yes":
    correct =  "Linear baseline correction"
    initialLin=[-10,0]
    def LinCorr(x,*p):
        total = len(x)
        # arrondi nombre de points
        zone = int(np.floor(total*fraction))
        #print("zone", type(zone))
        lincorr=np.ones(zone)
        j=0
        for i in x:
            if j < zone : 
                lincorr[j]=p[1]*i+p[0]
            j=j+1
        return lincorr
else:
    correct =  "No baseline correction"



# model definition
# here we are force vs. indentation, so d0=p[2]/k

if model == "pyramid": # could use "or cone" ## usual one, same as JPK
    initialguess = [10**2,10**3,1]
    def Fit(x, *p):
        fit=np.ones(len(x))
        j=0
        for i in x:
            if i >= p[1]:
                fit[j]=p[2]+p[0]*(i-p[1])**2
            else:
                fit[j]=p[2]
            j=j+1
        return fit
elif model == "sphere": # simplified one, different from JPK
    initialguess = [10**2,10**3,1]
    def Fit(x, *p):
        fit=np.ones(len(x))
        j=0
        for i in x:
            if i >= p[1]:
                fit[j]=p[2]+p[0]*(i-p[1])**(3./2)
            else:
                fit[j]=p[2]
            j=j+1
        return fit
elif model == "cone": # similar JPK
    initialguess = [10**2,10**3,1]
    def Fit(x, *p):
        fit=np.ones(len(x))
        j=0
        for i in x:
            if i >= p[1]:
                fit[j]=p[2]+p[0]*(i-p[1])**2
            else:
                fit[j]=p[2]
            j=j+1
        return fit
elif model == "linear":
    initialguess = [10**(-2),10**3,1]
    def Fit(x, *p):
        fit=np.ones(len(x))
        j=0
        for i in x:
            if i >= p[1]:
                fit[j]=p[2]+p[0]*(i-p[1])
            else:
                fit[j]=p[2]
            j=j+1
        return fit
else :
    print("Error on model or not avalaible")
    
#---------------------------------------------------
# young modulus calculation
if model == "pyramid":# Bilodeau, 1992
    def ValYoung(p):
        young = np.round(p*(10**6)*(1.-eta**2)/(0.7453*np.tan(alpha*np.pi/180.)), decimals=1)
        return young
elif model == "sphere":# Hertz, 1881
    def ValYoung(p):
        young = np.round(p*(10**6)*3.*(1.-eta**2)/(4.*np.sqrt(1000.*Rb)), decimals=1) #radius in nm
        return young
if model == "cone":# Sneddon, 1965
    def ValYoung(p):
        young = np.round(p*(10**6)*(1.-eta**2)*np.pi/(2.*np.tan(alpha*np.pi/180.)), decimals=1)
        return young
if model == "linear": #krieg, 2009
    def ValYoung(p):
        tension=np.round(p*(10**3)/(4.*np.pi*(1.+Rb/Rc)), decimals=1) # 10**3=10**6 (to get microN/m) * 10**(-3) (pour k en mN/m)
        return tension
#else :
#    print "(Output of model is not avalaible yet)"
#---------------------------------------------------
# parameter name
if model == "linear":
    parametre = "Tension (microN/m)"
else:
    parametre = "Young modulus (Pa)"
#---------------------------------------------------

# print out the model and parameters
if quantification=='mechanics':
    print("Fitting with", model, "model")
    file.write("# Fitting with "+ model+ " model\n")
    if model == "pyramid":
        print("Half angle =", alpha, ' deg')
        file.write("# Half angle = "+ str(alpha)+ " deg\n")
    elif model == "cone":
        print("Half angle =", alpha, ' deg')
        file.write("# Half angle = "+ str(alpha)+ " deg\n")
    elif model == "sphere":
        print("Bead radius =", Rb, ' um')
        file.write("# Bead radius = "+ str(Rb)+ " um\n")
    elif model == "linear":
        print("Bead radius =", Rb, ' um')
        file.write("# Bead radius = "+ str(Rb)+ " um\n")
        print("Cell radius =", Rc, ' um')
        file.write("# Cell radius = "+ str(Rc)+ " um\n")
    print(correct)
elif quantification=='adhesion':
    print('Measuring max detachment forces')
print("---------------------------------------------------")
file.write("#---------------------------------------------------\n")
if correction == "yes":
    file.write("# Baseline correction : "+correction+ " over "+ str(100*fraction)+ " % of curve\n")
else:
    file.write("# Baseline correction : "+correction+"\n")
file.write("#---------------------------------------------------\n")

# preparation du fichier de sortie, separe par des espaces
if quantification=='mechanics':
    file.write("# Filename | "+parametre+" | [error] | Contact Point (nm) | [error] | RMS fit (pN)\n | Curve length (nm)")
elif quantification=='adhesion':
    file.write("# Filename | Adhesion (pN)\n")
file.write("#---------------------------------------------------\n")
#-------------------------------------------------------
# LOOPING OVER FILES !
#-------------------------------------------------------
for fichier in files:
    localfichier=inputpath+fichier
    #---------------------------------------------------
    # loading txt export TRACE ONLY files with all channels
    # h is the measured smoothed height
    # h2 is the measured height
    # h3 is the height : DO NOT USE
    # from JPK exported file
    # fancyNames: "Height (measured & smoothed)" "Vertical Deflection" "Height (measured)" "Height" "Lateral Deflection" "Series Time" "Segment Time"
    # ATTENTION POUR MAPPING IL Y A UNE COLONNE EN PLUS : error -> e    
    df1 = pd.read_csv(localfichier, delimiter=r"\s+",   comment='#', names=['h', 'f','h2', 'e', 'h3', 'l', 't', 't2'], skiprows=76)
    
    #---------------------------------------------------
    # definition of plot as a grid to have the residues above the fitted curve
    gs = gridspec.GridSpec(2, 1,height_ratios=[1,3])
    # for axis label localization, to align the labels
    labelx = -0.1
    #---------------------------------------------------
    
    # create main figure  
    fig1=plt.figure(fichier, figsize=(7, 7), dpi=100)
    #---------------------------------------------------
    

    #---------------------------------------------------
    # data preparation before plot and fit

    # returning the FC for being all the same as mechs
#    df2=df1.copy()
#    plt.plot(df1['h'], df1['f'], color='red', alpha=0.5)
#   if quantification =='adhesion':
#       df2=df1.iloc[::-1].reset_index(drop=True)
        
#    plt.plot(df2['h'], df2['f'], color='red', alpha=0.5)
#    # conversion to usual values
#    if quantification=='adhesion':
#            microns = df2['h']*10**6 # piezo en  microns
#            piconewtons = df2['f']*10**12# force en pN
#    else:
#        microns = df1['h']*10**6 # piezo en  microns
#        piconewtons = df1['f']*10**12# force en pN
    if quantification =='adhesion':
        df2=df1.iloc[::-1].reset_index(drop=True)
        microns = df2['h']*10**6 # piezo en  microns
        piconewtons = df2['f']*10**12 # force en pN
    else :
        microns = df1['h']*10**6 # piezo en  microns
        piconewtons = df1['f']*10**12 # force en pN
    
#    #TSS calculation with no correction on original data
#    #converted to nm
#    #carefull with sign (depends on global orientation of FC): check with force curve on glass
#    # NOTE : TSS et deflection en nm pour FIT
#    TSS = 1000*microns + piconewtons / k # k est en pN/nm
#    #non corrected deflection, in nm
#    #no baseline correction, no tilt correction for the moment
#    deflection = piconewtons / k
#    #IMPORTANT : to have only positive values, shift the min value to 0, for FIT
#    deflection = deflection - np.min(deflection)
#    piconewtons2 = piconewtons - np.min(piconewtons)
    
    #linear correction of baseline
    if correction == "yes":
        # creating subset for correction
        # arrondi nombre de points
        zone = int(np.floor(len(piconewtons)*fraction))
        subsetLin=np.ones(zone)
        a=0
        for i in piconewtons:
            if a < zone : 
                subsetLin[a]=piconewtons[a]
            a=a+1
        LinParams, LinCovariances = curve_fit(LinCorr, microns, subsetLin, p0=initialLin)
        #print LinParams
        piconewtons = piconewtons - (LinParams[0]+LinParams[1]*microns)
        #print LinParams[0], LinParams[1]
    #piconewtons = piconewtons# - np.mean(piconewtons)   #-np.min(piconewtons) # esthetique, mais refitte
    
    
    #TSS calculation AFTER correction on original data (cf. Felix, Juin 2016)
    #converted to nm
    #carefull with sign (depends on global orientation of FC): check with force curve on glass
    # NOTE : TSS et deflection en nm pour FIT
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
    
    #valeur max du TSS pour avoir apres l'indentation avec le point de contact
    maxX=np.max(x)
    
    #---------------------------------------------------
    
    # plot corrected for TSS
    #plt.plot(x, piconewtons2, color='blue', alpha=0.25) #non corrected for liln

    
    if quantification=='mechanics':
        
        gs = gridspec.GridSpec(2, 1,height_ratios=[1,3])
    # for axis label localization, to align the labels
        labelx = -0.1
    #---------------------------------------------------
    
    # create main figure  
        fig1=plt.figure(fichier, figsize=(7, 7), dpi=100)
        
            # first sub
        ax1 = plt.subplot(gs[1])
        plt.plot(x, piconewtons, color='red', alpha=0.5) #corr for lin
        # set the plot
        plt.ylabel('Pushing, Force (pN)', fontsize = 16)
        plt.xlabel('TSS (nm)', fontsize = 16)
        plt.xlim(Xplot,)
        #plt.ylim(Yplot,)
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
        # except of Inf due to fitting bad (hence replace by NaN)
        # calculate Young modulus and its error
        # on a le droit de directement transformer l'erreur car c'est lineaire 
        # entre facteur estime, p[0] du fit, et facteur voulu, E ou Tc.
        # determine the contact point and its error
    
        if np.isinf(np.sum(fitCovariances)):
            error = errorYoung = "NaN"
            Young = ValYoung(fitParams[0])
            contact = np.round(fitParams[1], decimals=1)
            errorcontact = "NaN"
        else :
            error = np.sqrt(np.diag(fitCovariances))
            Young = ValYoung(fitParams[0])
            errorYoung = ValYoung(error[0])
            contact = np.round(fitParams[1], decimals=1)
            errorcontact = np.round(error[1], decimals=1)
        #---------------------------------------------------
    
        # output to command line
        print(fichier)
        if model == "pyramid" or "cone" or "sphere":
            print("Young modulus (Pa) = ", Young, " +/-", errorYoung)
        elif model == "linear":
            print("Tension (microN/m) = ", Young, " +/-", errorYoung)
        else:
            print("Not yet defined")
        #---------------------------------------------------
        # output to command line
        #print "Beginning of non contact zone is at zero displacement" #cf reorientation of FC above
        print("Contact point (nm) = ", contact, ' +/-', errorcontact)
        
        print("Curve length (nm) = ", maxX)
        
        
        # second subplot : residues
        ax2 = plt.subplot(gs[0], sharex=ax1)
        # calculate residues
        residus = piconewtons - fitted
        # plot residues of fits
        plt.plot(x, residus, "-", color="black", alpha=0.5)
        # set the plot
        plt.ylabel('Res. (pN)', fontsize = 16)
        plt.xlim(Xplot,)
        plt.ylim(-Yres,Yres)
        # align y axis label
        ax2.yaxis.set_label_coords(labelx, 0.5)
        # remove ticks of x axis
        plt.setp( ax2.get_xticklabels(), visible=False)
        #---------------------------------------------------
        
        # evaluation de la qualite du fit par RMS (cf JPK DP)
        N=len(fitted)
        carreresidus = residus**2
        rms=np.round(np.sqrt(carreresidus.sum()/N), decimals=1)
        print("RMS residuals (pN) = ", rms)
        print("---------------------------------------------------")
    
        
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
        
        texte=fichier+"\n"+"-------------\n"+"Model : "+model+"\n"+"Baseline corrected : "+correction+"\n"+"-------------\n"+ parametre +"\n"+ str(Young) + " +/- "+ str(errorYoung) + "\n" + "contact point (nm)\n" + str(contact) + ' +/- ' + str(errorcontact) + "\n" +"RMS residues (nm)\n"+str(rms) +'\n'+ "curve length (nm)\n"+str(maxX)+"\n"
        ax1.text(1000,100, texte, fontsize=10)  
        
        #---------------------------------------------------
        # output to text file
        file.write(fichier+" "+str(Young)+" "+str(errorYoung)+" "+str(contact)+" "+str(errorcontact)+" "+str(rms)+" "+ str(maxX)+"\n")
        #---------------------------------------------------    

    elif quantification =='adhesion':
        
        
        fig1=plt.figure(fichier, figsize=(7, 7), dpi=100)

        plt.plot(x, piconewtons, color='red', alpha=0.5)
        smoothX=x.rolling(window=smoothparam).mean().values
        smoothY=piconewtons.rolling(window=smoothparam).mean().values
        print('------------------------------')
        print(fichier)
        print("Adhesion=",pd.DataFrame(smoothY).min().values[0], "pN")
#        print(type(smoothX))
#        print(type(smoothY))
#        
        plt.plot(smoothX, smoothY, color="black")
        plt.xlabel('TSS (nm)')
        plt.ylabel('Pulling, Force (pN)')
#    
#        
#        minY=smoothY.min()
#        print(minY)
#        posminY=pd.Series(smoothY).index(minY)
#        adhesion=np.round(np.abs(minY), decimals=1)
#        adhesion=np.round(np.abs(piconewtons.min()), decimals=1)
#        print(adhesion)
#        plt.scatter(x[piconewtons.argmin()], -adhesion, s=20, c='green', marker='o')
        adhesion=np.round(np.abs(pd.DataFrame(smoothY).min().values), decimals=1)
        plt.axhline(y=0, linestyle='--', linewidth=1)
        plt.axhline(y=-adhesion, linestyle='--', linewidth=1, color='green')
        
        # second subplot : residues
#        ax2 = plt.subplot(gs[0], sharex=ax1)
#        # calculate residues
#        residus = piconewtons - smoothY
#        # plot residues of fits
#        plt.plot(x, residus, "-", color="black", alpha=0.5)
#        # set the plot
#        plt.ylabel('Res. (pN)', fontsize = 16)
#        plt.xlim(Xplot,)
#        plt.ylim(-Yres,Yres)
#        # align y axis label
#        ax2.yaxis.set_label_coords(labelx, 0.5)
#        # remove ticks of x axis
#        plt.setp( ax2.get_xticklabels(), visible=False)
        
        file.write(fichier+" "+str(adhesion[0])+"\n")
        
    else :
        print("MASSIVE ERROR")

    # esthetical for the entire plot
    fig1.tight_layout() 

    # show graphs on screen
    # output to figure files
    plotname=outputpath+fichier+'.png'
    plt.savefig(plotname)
    #release memory
    plt.close(fig1)
    #---------------------------------------------------

file.close()
