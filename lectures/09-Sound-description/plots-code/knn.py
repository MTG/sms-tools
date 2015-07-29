import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os, sys
from numpy import random
from scipy.stats import mode

def eucDist(vec1, vec2):
  return np.sqrt(np.sum(np.power(np.array(vec1) - np.array(vec2), 2)))

n = 30
qn = 8
K = 3
class1 = np.transpose(np.array([np.random.normal(-2,2,n), np.random.normal(-2,2,n)]))
class2 = np.transpose(np.array([np.random.normal(2,2,n), np.random.normal(2,2,n)]))
query = np.transpose(np.array([np.random.normal(0,2,qn), np.random.normal(0,2,qn)]))

plt.figure(1, figsize=(9.5, 3.5))

plt.subplot(1,2,1)
plt.scatter(class1[:,0],class1[:,1], c='b', alpha=0.7, s=50, edgecolor='none')
plt.scatter(class2[:,0],class2[:,1], c='r', alpha=0.7, s=50, edgecolor='none')
plt.scatter(query[:,0],query[:,1], c='c', alpha=1, s=50)

predClass = []
for kk in range(query.shape[0]):
    dist = []
    for pp in range(class1.shape[0]):
        euc = eucDist(query[kk,:], class1[pp,:])
        dist.append([euc, 1])
    
    for pp in range(class2.shape[0]):
        euc = eucDist(query[kk,:], class2[pp,:])
        dist.append([euc, 2])

    dist = np.array(dist)
    indSort = np.argsort(dist[:,0])
    topKDist = dist[indSort[:K],1]
    predClass.append(mode(topKDist)[0][0].tolist()) 

predClass = np.array(predClass)
indC1 = np.where(predClass==1)[0]
indC2 = np.where(predClass==2)[0]

plt.subplot(1,2,2)


plt.scatter(class1[:,0],class1[:,1], c='b', alpha=0.3, s=50, edgecolor='none')
plt.scatter(class2[:,0],class2[:,1], c='r', alpha=0.3, s=50, edgecolor='none')
plt.scatter(query[indC1,0],query[indC1,1], c='b', alpha=1, s=50)
plt.scatter(query[indC2,0],query[indC2,1], c='r', alpha=1, s=50)


plt.tight_layout()
plt.savefig('knn.png')
plt.show()
