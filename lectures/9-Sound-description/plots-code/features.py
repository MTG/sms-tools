import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os, sys
import json
from scipy.cluster.vq import vq, kmeans, whiten

def fetchDataDetails(inputDir, descExt = '.json'):
  dataDetails = {}
  for path, dname, fnames  in os.walk(inputDir):
    for fname in fnames:
      if descExt in fname.lower():
        rname, cname, sname = path.split('/')
        if not dataDetails.has_key(cname):
          dataDetails[cname]={}
        fDict = json.load(open(os.path.join(rname, cname, sname, fname),'r'))
        dataDetails[cname][sname]={'file': fname, 'feature':fDict}
  return dataDetails

def plotFeatures(inputDir, descInput = ('',''), anotOn =0):
  #mfcc descriptors are an special case for us as its a vector not a value
  descriptors = ['', '']
  mfccInd = [-1 , -1]
  if "mfcc" in descInput[0]:
    featType, featName, stats, ind  = descInput[0].split('.')
    descriptors[0] = featType+'.'+featName+'.'+stats
    mfccInd[0] = int(ind)
  else:
    descriptors[0] = descInput[0]

  if "mfcc" in descInput[1]:
    featType, featName, stats, ind  = descInput[1].split('.')
    descriptors[1] = featType+'.'+featName+'.'+stats
    mfccInd[1] = int(ind)
  else:
    descriptors[1] = descInput[1]

  dataDetails = fetchDataDetails(inputDir)
  colors = ['r', 'g', 'c', 'b', 'k', 'm', 'y']
  plt.figure(1, figsize=(9.5, 6))
  plt.hold(True)
  legArray = []
  catArray = []
  for ii, category in enumerate(dataDetails.keys()):
    catArray.append(category)
    for soundId in dataDetails[category].keys():
      filepath = os.path.join(inputDir, category, soundId, dataDetails[category][soundId]['file'])
      descSound = json.load(open(filepath, 'r'))
      if not descSound.has_key(descriptors[0]) or not descSound.has_key(descriptors[1]):
          print "Please provide descriptors which are extracted and saved before"
          return -1
      if "mfcc" in descriptors[0]:
        x_cord = descSound[descriptors[0]][0][mfccInd[0]]
      else:
        x_cord = descSound[descriptors[0]][0]

      if "mfcc" in descriptors[1]:
        y_cord = descSound[descriptors[1]][0][mfccInd[1]]
      else:
        y_cord = descSound[descriptors[1]][0]

      plt.scatter(x_cord,y_cord, c = colors[ii], s=50, hold = True, alpha=0.75)
      if anotOn==1:
         plt.annotate(soundId, xy=(x_cord, y_cord), xytext=(x_cord, y_cord))
    
    circ = Line2D([0], [0], linestyle="none", marker="o", alpha=0.75, markersize=10, markerfacecolor=colors[ii])
    legArray.append(circ)
  
  plt.ylabel(descInput[1], fontsize =16)
  plt.xlabel(descInput[0], fontsize =16)
  plt.legend(legArray, catArray ,numpoints=1,bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=len(catArray), mode="expand", borderaxespad=0.)

  plt.savefig('features.png')
  plt.show()

  

########################

plotFeatures('freesound-sounds', descInput = ('lowlevel.spectral_centroid.mean','lowlevel.mfcc.mean.2'), anotOn =0)
  
  
  
  
  
      
    
    
    
  
