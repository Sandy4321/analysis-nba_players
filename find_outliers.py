# ---------------------------------------------------------
# Outlier detection by commute-time and Euclidean distances
# 
# Sercan Taha Ahi, Nov 2011 (tahaahi at gmail dot com)
# ---------------------------------------------------------
import numpy as np
import scipy as Sci
from scipy import linalg
from scipy.cluster.vq import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mm
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import pylab
import sys
#import math
import string
import nnmf
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
    
    flabels = ["POS", "AGE", "G", "MINPG", "s1MPG", "s1APG", "s2MPG", "s2APG", "s3MPG", "s3APG", "ORBPG", "DRBPG", "ASTPG", "STLPG", "BLKPG", "TOPG", "PFPG"]
    X = np.hstack([POS, AGE, G, MINPG, s1MPG, s1APG, s2MPG, s2APG, s3MPG, s3APG, ORBPG, DRBPG, ASTPG, STLPG, BLKPG, TOPG, PFPG])
    print "\tX: " + str(X.shape[0]) + "x" + str(X.shape[1])
    
    ptr1 = np.array(X[:,0]==1).transpose()
    ptr2 = np.array(X[:,0]==2).transpose()
    ptr3 = np.array(X[:,0]==3).transpose()
    
    return X, flabels, PLY, ptr1, ptr2, ptr3
    

def get_stats(X, ptr1, ptr2, ptr3, PLY, idx, flabels):
    plt.figure()
    ax1 = plt.axes()
    bp = plt.boxplot(X, whis=1.5)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')
    xtickNames = plt.setp(ax1, xticklabels=flabels)
    plt.setp(xtickNames, rotation=45, fontsize=8)
    plt.draw()
    plt.savefig("boxplot.png", dpi=300)
    
    # (PCA)
    # pos = {C, F, G}
    p = mm.PCA(X[:,1:])
    print "\nPCA:"
    print p.fracs
    print sum(p.fracs)
    
    xmin = min(p.Y[:,0])
    xmax = max(p.Y[:,0])
    xrng = xmax - xmin
    ymin = min(p.Y[:,1])
    ymax = max(p.Y[:,1])
    yrng = ymax - ymin
    fx = 8
    fy = yrng * fx / xrng
    plt.figure(figsize=(fx,fy))
    '''
    plt.plot(p.Y[ptr1,0], np.zeros((sum(ptr1),1))+1, 'ro', hold='on')
    plt.plot(p.Y[ptr2,0], np.zeros((sum(ptr2),1))+2, 'go', hold='on')
    plt.plot(p.Y[ptr3,0], np.zeros((sum(ptr3),1))+3, 'bo', hold='on')
    plt.ylim(0,4)
    '''
    plt.plot(p.Y[ptr1,0], p.Y[ptr1,1], 'ro', hold='on')
    plt.plot(p.Y[ptr2,0], p.Y[ptr2,1], 'go', hold='on')
    plt.plot(p.Y[ptr3,0], p.Y[ptr3,1], 'bo', hold='on')
    '''
    for i in range(10):
        print idx[i]
        if sum(nonzero(ptr1)==idx[i])>0:
            plt.plot(p.Y[idx[i],0], p.Y[idx[i],1], 'rs', hold='on')
        elif sum(nonzero(ptr2)==idx[i])>0:
            plt.plot(p.Y[idx[i],0], p.Y[idx[i],1], 'gs', hold='on')
        elif sum(nonzero(ptr3)==idx[i])>0:
            plt.plot(p.Y[idx[i],0], p.Y[idx[i],1], 'bs', hold='on')
    '''
    plt.draw()
    plt.savefig("PCA_out1.png", dpi=300)
    return
    
    
def normalize(X):
    n_samples = X.shape[0]
    n_features = X.shape[1]
    print "\tn_samples  = " + str(n_samples)
    print "\tn_features = " + str(n_features)
    
    Xmean = X.mean(axis=0)
    Xstd  = X.std(axis=0)
    
    Y = np.zeros((n_samples, n_features))
    for i in range(0,n_samples):
        Y[i,:] = X[i,:] - Xmean
        Y[i,:] = Y[i,:] / Xstd
        
    #for i in range(0,n_features):
    #    print "\t\t" + str(Y[:,i].std())
    #print "\tmax = " + str(Y.max())
    #print "\tmin = " + str(Y.min())
    return Y

    
def dist_mahalanobis(X):
    n_samples = X.shape[0]
    n_features = X.shape[1]
    
    Xmean = X.mean(axis=0)
    Xc = np.zeros((n_samples, n_features))
    for i in range(0,n_samples):
        Xc[i,:] = X[i,:] - Xmean
    
    S = np.dot(Xc.T, Xc) / (n_features-1)
    Si = linalg.inv(S)
    
    D = np.zeros((n_samples, n_samples))
    for i in range(0,n_samples):
        a = X[i,:]
        a.reshape(1,n_features)
        for j in range(0,n_samples):
            b = X[j,:]
            b.reshape(1,n_features)
            D[i,j] = np.dot(np.dot(a-b,Si),(a-b).T)
    
    return D

    
def dist_euclidean(X):
    n_samples = X.shape[0]
    #n_features = X.shape[1]
    
    D = np.zeros((n_samples, n_samples))
    for i in range(0,n_samples):
        a = X[i,:]
        for j in range(0,n_samples):
            b = X[j,:]
            D[i,j] = linalg.norm(a-b)
    
    return D

    
def dist_commute_time(X):
    n_samples = X.shape[0]
    #n_features = X.shape[1]
    
    E = dist_euclidean(X)
    Estd = E.std()
    A = np.exp(-E**2 / Estd**2);
    D = np.zeros((n_samples,n_samples))
    for i in range(0,n_samples):
        A[i,i] = 0
        D[i,i] = A[i,:].sum(dtype=float)
    
    V = A.sum(dtype=float)
    print "\tGraph volume = " + str(V)
    
    #D = diag(A.sum(axis=1));
    L = D - A;
    Lp = linalg.pinv(L);
    
    CTD = np.zeros((n_samples,n_samples))
    for i in range(0,n_samples):
        for j in range(0,n_samples):
            CTD[i,j] = V * (Lp[i,i] + Lp[j,j] - 2*Lp[i,j])
    
    #CTD = CTD / CTD.max()
    #E = E / E.max()
    
    return CTD, E
    
    
def get_top_n_outliers(D, knn, n, dtype):
    n_samples = D.shape[0]
    
    Dtop = np.zeros((n_samples))
    for i in range(0,n_samples):
        idx = np.argsort(D[i,:])
        idx = idx[1:knn+1]
        Dtop[i] = D[i,idx].mean()
    
    idx = np.argsort(-Dtop)
    idx = idx[0:n]
    
    fname = dtype + ".png"
    x = np.arange(n_samples)
    y = np.sort(Dtop)
    plt.figure()
    plt.plot(x, y, 'b-')
    plt.grid()
    plt.title(dtype + " Distance")
    plt.xlabel("Player Index")
    plt.ylabel("Distance")
    plt.draw()
    plt.savefig(fname, dpi=300)
    
    return idx
    
    
def main():
    if len(sys.argv) == 1:
        target_year = 1997
    else:
        target_year = int(sys.argv[1])
    fname = str(target_year) + '-' + str(target_year+1) + '.csv'
    
    X, flabels, PLY, ptr1, ptr2, ptr3 = parse_stats(fname)
    X = normalize(X)
    CTD, E = dist_commute_time(X)
    M = dist_mahalanobis(X)
    
    knn = 10
    n = 10
    
    print "\nOutliers based on commute time distance:"
    dtype = "Commute Time"
    idx1 = get_top_n_outliers(CTD, knn, n, dtype)
    for i in range(0,n):
        print str(i+1).rjust(3) + "\t" + PLY[idx1[i]]
    
    print "\nOutliers based on Euclidean distance:"
    dtype = "Euclidean"
    idx2 = get_top_n_outliers(E, knn, n, dtype)
    for i in range(0,n):
        print str(i+1).rjust(3) + "\t" + PLY[idx2[i]]
    
    print "\nOutliers based on Mahalanobis distance:"
    dtype = "Mahalanobis"
    idx3 = get_top_n_outliers(M, knn, n, dtype)
    for i in range(0,n):
        print str(i+1).rjust(3) + "\t" + PLY[idx3[i]]
        
    get_stats(X, ptr1, ptr2, ptr3, PLY, idx1, flabels)
    
    return
    
    
def kmeans_test():
    # generate some random xy points and
    # give them some striation so there will be "real" groups.
    xy = np.random.rand(30,2)
    xy[3:8,1] -= .9
    xy[22:28,1] += .9
    
    # make some z vlues
    z = np.sin(xy[:,1]-0.2*xy[:,1])
    
    # whiten them
    z = whiten(z)
    
    # let scipy do its magic (k==3 groups)
    #res, idx = kmeans2(np.array(zip(xy[:,0],xy[:,1],z)), 3)
    res, idx = kmeans2(xy, 3)
    
    # convert groups to rbg 3-tuples.
    colors = ([([0,0,0],[1,0,0],[0,0,1])[i] for i in idx])
    
    # show sizes and colors. each color belongs in diff cluster.
    pylab.scatter(xy[:,0], xy[:,1], c=colors)
    pylab.savefig('clust.png')
    #print idx
    return
    
    
def nnmf_analysis():
    if len(sys.argv) == 1:
        target_year = 1997
    else:
        target_year = int(sys.argv[1])
    fname = str(target_year) + '-' + str(target_year+1) + '.xls'
    
    X, flabels, PLY, ptr1, ptr2, ptr3 = parse_stats(fname)
    # pos = {C, F, G}
    ptr1 = np.array(X[:,0]==1).transpose()
    ptr2 = np.array(X[:,0]==2).transpose()
    ptr3 = np.array(X[:,0]==3).transpose()
    X = normalize(X)
    
    # NNMF
    w, h = nnmf.factorize(X[:,1:], pc=3, iter=100)
    print np.shape(h)
    print np.shape(w)
    print
    print h[:,0]
    print
    print h[:,1]
    print
    print h[:,2]
    print
    print h[:,3]
    
    return
    
    
def kmeans_players():
    if len(sys.argv) == 1:
        target_year = 1997
    else:
        target_year = int(sys.argv[1])
    fname = str(target_year) + '-' + str(target_year+1) + '.xls'
    
    X, flabels, PLY, ptr1, ptr2, ptr3 = parse_stats(fname)
    # pos = {C, F, G}
    ptr1 = np.array(X[:,0]==1).transpose()
    ptr2 = np.array(X[:,0]==2).transpose()
    ptr3 = np.array(X[:,0]==3).transpose()
    X = normalize(X)
    
    res, idx = kmeans2(X[:,1:], 3)
    colors = ([([1,0,0],[0,1,0],[0,0,1])[i] for i in idx])
    #print ptr1.shape
    
    # (PCA)
    # pos = {C, F, G}
    p = mm.PCA(X[:,1:])
    xmin = min(p.Y[:,0])
    xmax = max(p.Y[:,0])
    xrng = xmax - xmin
    ymin = min(p.Y[:,1])
    ymax = max(p.Y[:,1])
    yrng = ymax - ymin
    fx = 8
    fy = yrng * fx / xrng
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    
    for n in np.nonzero(ptr1)[0].tolist():
        ax.text(p.Y[n,0], p.Y[n,1], p.Y[n,2], str(n), color='r', size=8)
        ax.hold(True)
        
    for n in np.nonzero(ptr2)[0].tolist():
        ax.text(p.Y[n,0], p.Y[n,1], p.Y[n,2], str(n), color='g', size=8)
        ax.hold(True)
        
    for n in np.nonzero(ptr3)[0].tolist():
        ax.text(p.Y[n,0], p.Y[n,1], p.Y[n,2], str(n), color='b', size=8)
        ax.hold(True)
        
    ax.set_xlim(p.Y[:,0].min(), p.Y[:,0].max())
    ax.set_ylim(p.Y[:,1].min(), p.Y[:,1].max())
    ax.set_zlim(p.Y[:,2].min(), p.Y[:,2].max())
    
    #ax.plot(p.Y[ptr1,0], p.Y[ptr1,1], p.Y[ptr1,2], 'ro'); ax.hold(True)
    #ax.plot(p.Y[ptr2,0], p.Y[ptr2,1], p.Y[ptr2,2], 'go'); ax.hold(True)
    #ax.plot(p.Y[ptr3,0], p.Y[ptr3,1], p.Y[ptr3,2], 'bo'); ax.hold(True)
    ax.view_init(30, 60)
    plt.draw()
    plt.savefig("pca_3d.png", dpi=100)
    '''
    for angle in range(90):
        ax.view_init(30, angle)
        #ax.view_init(angle, 60)        
        plt.draw()
        fname = "pca_3d_" + str(angle).zfill(3) + ".png"
        plt.savefig(fname, dpi=100)
    '''
    print
    print '\t# of centers  = ' + str(sum(ptr1))
    print '\t# of forwards = ' + str(sum(ptr2))
    print '\t# of guards   = ' + str(sum(ptr3))
    print
    print '\t# of class1 = ' + str(sum(idx==0))
    print '\t# of class2 = ' + str(sum(idx==1))
    print '\t# of class3 = ' + str(sum(idx==2))
    
    #'''
    plt.figure(figsize=(fx,fy))
    for n in range(len(idx)):
        if ptr1[n]==True:
            plt.plot(p.Y[n,0], p.Y[n,1], 'o', c=colors[n], hold='on')
        elif ptr2[n]==True:
            plt.plot(p.Y[n,0], p.Y[n,1], 's', c=colors[n], hold='on')
        elif ptr3[n]==True:
            plt.plot(p.Y[n,0], p.Y[n,1], '^', c=colors[n], hold='on')
        else:
            print '\t!! ' + str(n)
    plt.draw()
    plt.savefig("kmeans_on_pca.png", dpi=300)
    #'''
    return
    
    
def scatterplot(data, data_name):
    '''
    Makes a scatterplot matrix:
    Inputs:
      data - a list of data [dataX, dataY,dataZ,...];
             all elements must have same length
 
      data_name - a list of descriptions of the data;
                  len(data) should be equal to len(data_name)
 
    Output:
      fig - matplotlib.figure.Figure Object
    '''
 
    N = len(data_name)
    print N
    fig = plt.figure()
    
    for i in xrange(N):
        for j in xrange(N):
            ax = fig.add_subplot(N, N, i*N+j+1)
 
            if j == 0: ax.set_ylabel(data_name[i], size='10')
            if i == 0: ax.set_title(data_name[j], size='10')
            if i == j:
                ax.hist(data[i], 10)
            else:
                ax.scatter(data[j], data[i])
 
    return fig
    
    
def demo_scatter_matrix_plot():
    target_year = 1997
    fname = str(target_year) + '-' + str(target_year+1) + '.csv'
    X, flabels, PLY, ptr1, ptr2, ptr3 = parse_stats(fname)
    
    #fig = scatterplot(X, flabels)
    #fig.savefig('scatterplot.png', dpi=120)
    #plt.show()
    return
    
    
#main()
#kmeans_players()
#nnmf_analysis()

demo_scatter_matrix_plot()