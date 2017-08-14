import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os, sys


def eucDist(vec1, vec2): 
  return np.sqrt(np.sum(np.power(np.array(vec1) - np.array(vec2), 2)))

vec1 = np.array([.3, .2])
vec2 = np.array([.6, .7])

plt.figure(1, figsize=(4, 3))

plt.scatter(vec1[0], vec1[1], c = 'r', s=50, alpha=0.75)
plt.scatter(vec2[0], vec2[1], c = 'b', s=50, alpha=0.75)
plt.plot([vec1[0], vec2[0]], [vec1[1], vec2[1]], 'k')
plt.ylabel('first dimension', fontsize =16)
plt.xlabel('second dimension', fontsize =16)


plt.tight_layout()
plt.savefig('euclidian-distance.png')
plt.show()
