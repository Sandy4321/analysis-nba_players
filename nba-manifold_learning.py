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
    print "\tX: " + str(X.shape[0]) + "x" + str(X.shape[1])
    
    return X, PLY
    
    
def demo1():
    if len(sys.argv) == 1:
        target_year = 1997
    else:
        target_year = int(sys.argv[1])
    fname = str(target_year) + '-' + str(target_year+1) + '.csv'
    
    X, PLY = parse_stats(fname)
    # pos = {C, F, G}
    ptr1 = np.array(X[:,0]==1).transpose()
    ptr2 = np.array(X[:,0]==2).transpose()
    ptr3 = np.array(X[:,0]==3).transpose()
    y = X[:,0]
    X = X[:,1:]
    return X, y
    
    
#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    colors = np.array([[1,0,0],[0,1,0],[0,0,1]])
    
    pl.figure()
    ax = pl.subplot(111)
    for n in xrange(X.shape[0]):
        pl.text(X[n, 0], X[n, 1], str(y[n]),
                color = colors[y[n]-1,:],
                fontdict = {'weight': 'bold', 'size': 9})
    
    pl.xticks([]), pl.yticks([])
    if title is not None:
        pl.title(title)
    return
    
    
X, y = demo1()
n_samples, n_features = X.shape
n_neighbors = 8


#----------------------------------------------------------------------
# Projection on to the first 2 principal components
# 
print "Computing PCA projection"
t0 = time()
X_pca = decomposition.RandomizedPCA(n_components=2).fit_transform(X)
plot_embedding(X_pca, "Principal Components projection of the players (time %.2fs)" % (time() - t0))

'''
#----------------------------------------------------------------------
# Random 2D projection using a random unitary matrix
# 
print "Computing random projection"
rng = np.random.RandomState(42)
Q, _ = qr_economic(rng.normal(size=(n_features, 2)))
X_projected = np.dot(Q.T, X.T).T
plot_embedding(X_projected, "Random Projection of the players")


#----------------------------------------------------------------------
# Projection on to the first 2 linear discriminant components
# 
print "Computing LDA projection"
X2 = X.copy()
X2.flat[::X.shape[1] + 1] += 0.01  # Make X invertible
t0 = time()
X_lda = lda.LDA(n_components=2).fit_transform(X2, y)
plot_embedding(X_lda, "Linear Discriminant projection of the players (time %.2fs)" % (time() - t0))


#----------------------------------------------------------------------
# Isomap projection of the players dataset
# 
print "Computing Isomap embedding"
n_components=2
t0 = time()
X_iso = manifold.Isomap(n_neighbors, n_components).fit_transform(X)
print "Done."
plot_embedding(X_iso, "Isomap projection of the players (time %.2fs)" % (time() - t0))


#----------------------------------------------------------------------
# Locally linear embedding of the players dataset
# 
print "Computing LLE embedding"
n_components=2
clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components, method='standard')
t0 = time()
X_lle = clf.fit_transform(X)
print "Done. Reconstruction error: %g" % clf.reconstruction_error_
plot_embedding(X_lle, "Locally Linear Embedding of the players (time %.2fs)" % (time() - t0))


#----------------------------------------------------------------------
# Modified Locally linear embedding of the players dataset
# 
print "Computing modified LLE embedding"
n_components=2
clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components, method='modified')
t0 = time()
X_mlle = clf.fit_transform(X)
print "Done. Reconstruction error: %g" % clf.reconstruction_error_
plot_embedding(X_mlle, "Modified Locally Linear Embedding of the players (time %.2fs)" % (time() - t0))


#----------------------------------------------------------------------
# HLLE embedding of the players dataset
# 
print "Computing Hessian LLE embedding"
n_components = 2
clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components, method='hessian')
t0 = time()
X_hlle = clf.fit_transform(X)
print "Done. Reconstruction error: %g" % clf.reconstruction_error_
plot_embedding(X_hlle, "Hessian Locally Linear Embedding of the players (time %.2fs)" % (time() - t0))


#----------------------------------------------------------------------
# LTSA embedding of the players dataset
# 
print "Computing LTSA embedding"
n_components = 2
clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components, method='ltsa')
t0 = time()
X_ltsa = clf.fit_transform(X)
print "Done. Reconstruction error: %g" % clf.reconstruction_error_
plot_embedding(X_ltsa, "Local Tangent Space Alignment of the players (time %.2fs)" % (time() - t0))
'''

pl.show()

