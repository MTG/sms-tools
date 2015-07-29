import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os, sys
from scipy.cluster.vq import vq, kmeans, whiten
from numpy import random
import pickle

n = 30
features = np.hstack((np.array([np.random.normal(-2,1.1,n), np.random.normal(-2,1.1,n)]), np.array([np.random.normal(2,1.5,n), np.random.normal(2,1.5,n)])))
whitened = np.transpose(features)
nClusters = 2
arr = np.arange(whitened.shape[0])
np.random.shuffle(arr)
seeds = np.array([[-2, 1], [2, -1]])
color = [ 'r', 'c', 'c', 'm']

plt.figure(1, figsize=(9.5, 4))

plt.subplot(1,3,1)
plt.scatter(whitened[:,0],whitened[:,1], c='b', alpha=0.75, s=50, edgecolor='none')


plt.subplot(1,3,2)
clusResults = -1*np.ones(whitened.shape[0])
for ii in range(whitened.shape[0]):
    diff = seeds - whitened[ii,:]
    diff = np.sum(np.power(diff,2), axis = 1)
    indMin = np.argmin(diff)
    clusResults[ii] = indMin

for pp in range(nClusters):
    plt.scatter(whitened[clusResults==pp,0],whitened[clusResults==pp,1], c=color[pp], alpha=0.75, s=50, edgecolor='none')
plt.scatter(seeds[:,0],seeds[:,1], c=color[:nClusters], alpha=1, s=80)

plt.subplot(1,3,3)
centroids, distortion = kmeans(whitened, seeds, iter=40)
clusResults = -1*np.ones(whitened.shape[0])

for ii in range(whitened.shape[0]):
    diff = centroids - whitened[ii,:]
    diff = np.sum(np.power(diff,2), axis = 1)
    indMin = np.argmin(diff)
    clusResults[ii] = indMin

for pp in range(nClusters):
    plt.scatter(whitened[clusResults==pp,0],whitened[clusResults==pp,1], c=color[pp], s=50, alpha=0.75, edgecolor='none')
plt.scatter(centroids[:,0],centroids[:,1], c=color[:nClusters], alpha=1, s=80)


plt.tight_layout()
plt.savefig('k-means.png')
plt.show()
