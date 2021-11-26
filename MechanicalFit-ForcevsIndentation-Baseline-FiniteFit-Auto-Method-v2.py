# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 13:44:24 2015
Corrected on mardi 7 juin 2016, 09:51:35 by PHP

@author: php
"""
#---------------------------------------------------
# ATTENTION : IL FAUT QUE LES COURBES SOIENT TRIEES ET CALIBREES
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
print now
print "---------------------------------------------------"
#---------------------------------------------------

# outlook for plots
plt.rcParams.update({'font.size': 16})
#plt.rcParams.update({'font.family':'serif'})

#---------------------------------------------------

# where is the file
#methode de traitement des donnees
methodpath = '/home/php/Bureau/test/'
methofilename = 'methodcut.txt'
methodloc= methodpath+methofilename
# data for processing
inputpath = '/home/php/Bureau/test/fc/'
outputpath = inputpath + maintenant + '-results/'
datasave = outputpath+maintenant +'-data.txt'
#on peut aussi choisir de sauver ailleurs...

# pour le zoom sur les courbes, voici le minimum
Xplot=4000 # 0 par defaut


#---------------------------------------------------

methode = open(methodloc, 'r')
methodelines = methode.read().splitlines() # avoid the \n at the end of the file
#print methodelines
methode.close()
# model name : pyramid, sphere, linear (for tension), cone
# LACKS : CYLINDER
model = methodelines[1]
print model
# CORRECTION baseline : yes, no ?
correction = methodelines[3]
# on what fraction ?
fraction = float(methodelines[5])
# ALL : corrected & used lever spring constant
k = float(methodelines[7]) #pN/nm
# PYR / CONE : half angle (to face for pyr) in degrees
alpha = float(methodelines[9])
# SPHERE / LIN : radius of bead in um
Rb = float(methodelines[11])
# LIN : radius of cell in um
Rc = float(methodelines[13])
# Finite indentation
finite = methodelines[15]
# wanted indentation
indentation = float(methodelines[17])
#---------------------------------------------------

# compressibilty : non compressible is 0.5
eta = 0.5

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


# find closer index value from a file
def Index(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

# linear correction of baseline on x% of FC

if correction == "yes":
    correct =  "Linear baseline correction"
    initialLin=[-10,0]
    def LinCorr(x,*p):
        total = len(x)
        # arrondi nombre de points
        zone = np.floor(total*fraction)
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
                #fit[j]=p[2]+p[0]*(i-p[1]+p[2]/k)**2
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
                #fit[j]=p[2]+p[0]*(i-p[1]+p[2]/k)**(3./2)
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
                #fit[j]=p[2]+p[0]*(i-p[1]+p[2]/k)**2
                fit[j]=p[2]+p[0]*(i-p[1]+)**2
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
                #fit[j]=p[2]+p[0]*(i-p[1]+p[2]/k)
                fit[j]=p[2]+p[0]*(i-p[1])
            else:
                fit[j]=p[2]
            j=j+1
        return fit
else :
    print "Error on model or not avalaible"
    
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
print "Fitting with", model, "model"
file.write("# Fitting with "+ model+ " model\n")
if model == "pyramid":
    print "Half angle =", alpha, ' deg'
    file.write("# Half angle = "+ str(alpha)+ " deg\n")
elif model == "cone":
    print "Half angle =", alpha, ' deg'
    file.write("# Half angle = "+ str(alpha)+ " deg\n")
elif model == "sphere":
    print "Bead radius =", Rb, ' um'
    file.write("# Bead radius = "+ str(Rb)+ " um\n")
elif model == "linear":
    print "Bead radius =", Rb, ' um'
    file.write("# Bead radius = "+ str(Rb)+ " um\n")
    print "Cell radius =", Rc, ' um'
    file.write("# Cell radius = "+ str(Rc)+ " um\n")
print correct
print "---------------------------------------------------"
file.write("#---------------------------------------------------\n")
if correction == "yes":
    file.write("# Baseline correction : "+correction+ " over "+ str(100*fraction)+ " % of curve\n")
else:
    file.write("# Baseline correction : "+correction+"\n")
file.write("#---------------------------------------------------\n")
if finite == "yes":
    file.write("# Indentation over first "+str(indentation)+ " nm\n")
else:
    file.write("# Indentation over the entire curve\n")
file.write("#---------------------------------------------------\n")

# preparation du fichier de sortie, separe par des espaces
file.write("# Filename | "+parametre+" | [error] | Contact Point (nm) | [error] | RMS fit (pN)\n")
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
    df1 = pd.read_csv(localfichier, delimiter=r"\s+",   comment='#', names=['h', 'f','h2', 'h3','h4', 'h5','l', 't', 't2'], skiprows=74)
    
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
    microns = df1['h']*10**6 # piezo en  microns
    piconewtons = df1['f']*10**12# force en pN
#    #TSS calculation with no correction on original data
#    #converted to nm
#    #carefull with sign (depends on global orientation of FC): check with force curve on glass
#    # NOTE : TSS et deflection en nm pour FIT
#    TSS = 1000*microns + piconewtons / k # k est en pN/nm
#    #non corrected deflection, in nm
#    #no baseline correction, no tilt correction for the moment
#    #deflection = piconewtons / k
#    #IMPORTANT : to have only positive values, shift the min value to 0, for FIT
#    #deflection = deflection - np.min(deflection)
#    #piconewtons2 = piconewtons - np.min(piconewtons)
    
    #linear correction of baseline
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
        #print LinParams[0], LinParams[1]
    piconewtons = piconewtons - np.min(piconewtons)   
    
    #TSS calculation with linear correction on original data (Felix 06.2016)
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
    
    #---------------------------------------------------
    
    # plot corrected for TSS
    #plt.plot(x, piconewtons2, '-', color='blue', alpha=0.25) #non corrected for liln
    plt.plot(x, piconewtons, '-', color='red', alpha=0.5) #corr for lin
    # set the plot
    plt.ylabel('F (pN)', fontsize = 16)
    plt.xlabel('Z-d (nm)', fontsize = 16)
    plt.xlim(Xplot,)
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
    print fichier
    if model == "pyramid" or "cone" or "sphere":
        print "Young modulus (Pa) = ", Young, " +/-", errorYoung
    elif model == "linear":
        print "Tension (microN/m) = ", Young, " +/-", errorYoung
    else:
        print "Not yet defined"
    #---------------------------------------------------
    # output to command line
    #print "Beginning of non contact zone is at zero displacement" #cf reorientation of FC above
    print "Contact point (nm) = ", contact, ' +/-', errorcontact
    
    Zc=fitParams[1]
    BL=fitParams[2]
    
    # ATTENTION...
    # fit on indentation only with Zc and BL guessed from all curve fitted
    # new model definition with local guesses
    if finite == "yes":

        
        if model == "pyramid": # could use "or cone" ## usual one, same as JPK
            initialguess2 = [10**2]
            def Fit2(x, *p):
                fit=np.ones(len(x))
                j=0
                for i in x:
                    if i >= Zc:
                        fit[j]=BL+p[0]*(i-Zc+BL)**2
                    else:
                        fit[j]=BL
                    j=j+1
                return fit
        elif model == "sphere": # simplified one, different from JPK
            initialguess2 = [10**2]
            def Fit2(x, *p):
                fit=np.ones(len(x))
                j=0
                for i in x:
                    if i >= Zc:
                        fit[j]=BL+p[0]*(i-Zc+BL)**(3./2)
                    else:
                        fit[j]=BL
                    j=j+1
                return fit
        elif model == "cone": # similar JPK
            initialguess2 = [10**2]
            def Fit2(x, *p):
                fit=np.ones(len(x))
                j=0
                for i in x:
                    if i >= Zc:
                        fit[j]=BL+p[0]*(i-Zc+BL)**2
                    else:
                        fit[j]=BL
                    j=j+1
                return fit
        elif model == "linear":
            initialguess2 = [10**(-2)]
            def Fit2(x, *p):
                fit=np.ones(len(x))
                j=0
                for i in x:
                    if i >= Zc:
                        fit[j]=BL+p[0]*(i-Zc+BL)
                    else:
                        fit[j]=BL
                    j=j+1
                return fit
        else :
            print "Error on model or not avalaible"
        
        # zone defition for new fit
        indexZc=Index(x, Zc)
        fin=Zc+indentation
        indexfin=Index(x, fin)
        final=len(x)-1 # attention sinon ca tourne en rond, s'il n'y a pas moins un
    
        subx=x[:indexfin]
        suby=piconewtons[:indexfin]
        # newfit
        fitParams2, fitCovariances2 = curve_fit(Fit2, subx, suby, p0=initialguess2)
        if np.isinf(np.sum(fitCovariances)):
            error = errorYoung = "NaN"
            Young = ValYoung(fitParams2[0])
        else :
            error2 = np.sqrt(np.diag(fitCovariances2))
            Young = ValYoung(fitParams2[0])
            errorYoung = ValYoung(error2[0])
        #recalculate data to superimpose the fit on the original curve
        fitted2 = Fit2(subx, fitParams2[0])
        # plot it
        plt.plot(subx, fitted2, color="black")
        
        # second subplot : residues
        ax2 = plt.subplot(gs[0], sharex=ax1)
        # calculate residues
        residus = suby - fitted2
        # plot residues of fits
        plt.plot(subx, residus, "-", color="black", alpha=0.5)
        # set the plot
        plt.ylabel('Res. (pN)', fontsize = 16)
        plt.xlim(Xplot,)
        plt.ylim(-25,25)
        # align y axis label
        ax2.yaxis.set_label_coords(labelx, 0.5)
        # remove ticks of x axis
        plt.setp( ax2.get_xticklabels(), visible=False)
        # evaluation de la qualite du fit par RMS (cf JPK DP)
        N=len(fitted2)
        carreresidus = residus**2
        rms=np.round(np.sqrt(carreresidus.sum()/N), decimals=1)
        print "RMS residuals (pN) = ", rms
        print "---------------------------------------------------"
        #---------------------------------------------------
    
    else:
        #---------------------------------------------------
        #recalculate data to superimpose the fit on the original curve
        fitted = Fit(x, fitParams[0], fitParams[1], fitParams[2])
        # plot it
        plt.plot(x, fitted, color="black")
                # second subplot : residues
        ax2 = plt.subplot(gs[0], sharex=ax1)
        # calculate residues
        residus = piconewtons - fitted
        # plot residues of fits
        plt.plot(x, residus, "-", color="black", alpha=0.5)
        # set the plot
        plt.ylabel('Res. (pN)', fontsize = 16)
        plt.xlim(Xplot,)
        plt.ylim(-25,25)
        # align y axis label
        ax2.yaxis.set_label_coords(labelx, 0.5)
        # remove ticks of x aerroxis
        plt.setp( ax2.get_xticklabels(), visible=False)
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
    
    # esthetical for the entire plot
    fig1.tight_layout() 
    # elegance
    
    texte=fichier+"\n"+"-------------\n"+"Model : "+model+"\n"+"Baseline corrected : "+correction+"\n"+"-------------\n"+ parametre +"\n"+ str(Young) + " +/- "+ str(errorYoung) + "\n" + "contact point (nm)\n" + str(contact) + ' +/- ' + str(errorcontact) + "\n" +"RMS residues (nm)\n"+str(rms)
    ax1.text(1000,100, texte, fontsize=10)  
    
    #---------------------------------------------------
    # output to text file
    file.write(fichier+" "+str(Young)+" "+str(errorYoung)+" "+str(Zc)+" "+str(errorcontact)+" "+str(rms)+"\n")
    #---------------------------------------------------    
    # show graphs on screen
    #plt.show()
    # output to figure files
    plotname=outputpath+fichier+'.png'
    plt.savefig(plotname)
    #release memory
    plt.close(fig1)
    #---------------------------------------------------

file.close()
