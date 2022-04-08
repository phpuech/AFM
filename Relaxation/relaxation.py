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
import seaborn as sns
#from skimage import data, io, filters

sns.set_style("white")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 1})

# where is the file
#methode de traitement des donnees
#methodpath = '/home/php/Documents/Data/Work/Manipes/2016/Marie-Julie/160503-mj-cell-processed/'
#methofilename = 'method2.txt' # ATTENTION C'EST LINEAIRE
#methodloc= methodpath+methofilename
# data for processing
inputpath = '/home/php/Bureau/'
fichier = 'test.txt'
localfichier = inputpath+fichier
#outputpath = inputpath + maintenant + '-results/'
#datasave = outputpath+maintenant +'-data.txt'
#on peut aussi choisir de sauver ailleurs...

df = pd.read_csv(localfichier, delimiter=r"\s+",   comment='#', names=['neg', 'pos'], skiprows=1)

plt.figure("Plot", figsize=(5,7))

sns.stripplot(data=df, jitter=True, color='white', edgecolor='black',   alpha=0.5)
sns.boxplot(data=df, saturation=0.75, width=0.35)

plt.show()