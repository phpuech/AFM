# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 15:15:12 2016

@author: php
"""

# Area under experimental curve

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

# where is the file
#methode de traitement des donnees
#methodpath = '/home/php/Documents/Data/Work/Manipes/2016/Marie-Julie/160503-mj-cell-processed/'
#methofilename = 'method2.txt' # ATTENTION C'EST LINEAIRE
#methodloc= methodpath+methofilename
# data for processing
inputpath = '/home/php/Bureau/Dev/PHP-DEV/TSS/test/'
fichier = 'essai.txt'
localfichier = inputpath+fichier
#outputpath = inputpath + maintenant + '-results/'
#datasave = outputpath+maintenant +'-data.txt'
#on peut aussi choisir de sauver ailleurs...

df = pd.read_csv(localfichier, delimiter=r"\s+",   comment='#', names=['h', 'f','h2', 'h3', 'h4', 'h5', 'l', 't', 't2'], skiprows=74)


k=11.

microns = df['h']*10**6 # piezo en  microns
piconewtons = df['f']*10**12# force en pN

TSS = 1000*microns + piconewtons / k

fig = plt.figure()

plt.plot(1000*microns, piconewtons, label='uncorrected')

plt.plot(TSS, piconewtons, label='corrected')

plt.legend()

plt.show()
