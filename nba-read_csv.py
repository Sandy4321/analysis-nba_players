import numpy as np
import scipy as Sci
import scipy.linalg
from scipy.cluster.vq import *
import matplotlib.pyplot as plt
import matplotlib.mlab as mm
import matplotlib.animation as animation
from matplotlib import offsetbox
from mpl_toolkits.mplot3d import Axes3D
import pylab as pl
import sys
import math
import string
from sklearn.utils.fixes import qr_economic
from sklearn import manifold, datasets, decomposition, lda
from time import time
import csv


def parse_stats(fname):
    ifile  = open(fname, "rb")
    reader = csv.reader(ifile)
    n_players = 0
    for row in reader: 
        n_players += 1
    n_players -= 1
    ifile.close()
    
    PLY  = list()   # [0]
    
    POS = np.zeros((n_players,1))   # [2]
    AGE = np.zeros((n_players,1))   # [6]
    G   = np.zeros((n_players,1))   # [7]
    MINPG = np.zeros((n_players,1)) # [9]
    
    s1MPG = np.zeros((n_players,1))
    s1APG = np.zeros((n_players,1))
    
    s2MPG = np.zeros((n_players,1))
    s2APG = np.zeros((n_players,1))
    
    s3MPG = np.zeros((n_players,1))
    s3APG = np.zeros((n_players,1))
    
    ORBPG = np.zeros((n_players,1))
    DRBPG = np.zeros((n_players,1))
    
    ASTPG = np.zeros((n_players,1))
    STLPG = np.zeros((n_players,1))
    BLKPG = np.zeros((n_players,1))
    TOPG  = np.zeros((n_players,1))
    PFPG  = np.zeros((n_players,1))
    
    ifile  = open(fname, "rb")
    reader = csv.reader(ifile)
    rownum = 0
    for row in reader:
        if (rownum > 0):
            colnum = 1
            for col in row:
                if (colnum==1):
                    PLY.append(col)
                elif (colnum==3):
                    if col=='C':
                        POS[rownum-1] = 1
                    elif col=='F':
                        POS[rownum-1] = 2
                    else:
                        POS[rownum-1] = 3
                elif (colnum==7):
                    AGE[rownum-1] = float(col)
                elif (colnum==8):
                    G[rownum-1] = float(col)
                elif (colnum==10):
                    MINPG[rownum-1] = float(col)
                elif (colnum==11):
                    PTS = float(col)
                elif (colnum==13):
                    FGM = float(col)
                elif (colnum==14):
                    FGA = float(col)
                elif (colnum==16):
                    s1MPG[rownum-1] = float(col) / G[rownum-1]
                elif (colnum==17):
                    s1APG[rownum-1] = float(col) / G[rownum-1]
                elif (colnum==19):
                    TPM = float(col)
                elif (colnum==20):
                    TPA = float(col)
                elif (colnum==23):
                    ORBPG[rownum-1] = float(col)
                elif (colnum==25):
                    DRBPG[rownum-1] = float(col)
                elif (colnum==29):
                    ASTPG[rownum-1] = float(col)
                elif (colnum==31):
                    STLPG[rownum-1] = float(col)
                elif (colnum==33):
                    BLKPG[rownum-1] = float(col)
                elif (colnum==35):
                    TOPG[rownum-1] = float(col)
                elif (colnum==37):
                    PFPG[rownum-1] = float(col)
                colnum += 1
                
            s3MPG[rownum-1] = TPM / G[rownum-1]
            s3APG[rownum-1] = TPA / G[rownum-1]
            s2MPG[rownum-1] = (FGM-TPM) / G[rownum-1]
            s2APG[rownum-1] = (FGA-TPA) / G[rownum-1]
        rownum += 1
    ifile.close()
    
    X = np.hstack([POS, AGE, G, MINPG, s1MPG, s1APG, s2MPG, s2APG, s3MPG, s3APG, ORBPG, DRBPG, ASTPG, STLPG, BLKPG, TOPG, PFPG])
    
    print X.shape
    return X
    
    
def demo1():
    if len(sys.argv) == 1:
        target_year = 1997
    else:
        target_year = int(sys.argv[1])
    fname = str(target_year) + '-' + str(target_year+1) + '.csv'
    
    parse_stats(fname)
    return
    
    
demo1()