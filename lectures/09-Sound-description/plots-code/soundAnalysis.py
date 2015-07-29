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
  plt.figure()
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

      plt.scatter(x_cord,y_cord, c = colors[ii], s=200, hold = True, alpha=0.75)
      if anotOn==1:
         plt.annotate(soundId, xy=(x_cord, y_cord), xytext=(x_cord, y_cord))
    
    circ = Line2D([0], [0], linestyle="none", marker="o", alpha=0.75, markersize=10, markerfacecolor=colors[ii])
    legArray.append(circ)
  
  plt.ylabel(descInput[1], fontsize =16)
  plt.xlabel(descInput[0], fontsize =16)
  plt.legend(legArray, catArray ,numpoints=1,bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=len(catArray), mode="expand", borderaxespad=0.)

  plt.show()
  
  
def eucDistFeatures(ftrDict1, ftrDict2):
  
  f1 =  convFtrDict2List(ftrDict1)
  f2 =  convFtrDict2List(ftrDict2)
  
  return eucDist(f1, f2)

def eucDist(vec1, vec2):
  
  return np.sqrt(np.sum(np.power(np.array(vec1) - np.array(vec2), 2)))

def convFtrDict2List(ftrDict):
  ftr = []
  for key in ftrDict.keys():
    if type(ftrDict[key][0]) == list:
      ftr.extend(ftrDict[key][0])
    else:
      ftr.extend(ftrDict[key])
      
  return ftr


def computeSimilarSounds(queryFile, targetDir):
  
  dataDetails = fetchDataDetails(targetDir)
  
  #reading query feature dictionary
  qFtr = json.load(open(queryFile, 'r'))
  
  dist = []
  #iterating over classes
  for cname in dataDetails.keys():
    #iterating over sounds
    for sname in dataDetails[cname].keys():
      eucDist = eucDistFeatures(qFtr, dataDetails[cname][sname]['feature'])
      dist.append([eucDist, sname, cname])
  
  #sorting the array based on the dist
  indSort = np.argsort(np.array(dist)[:,0])
  return (np.array(dist)[indSort,:]).tolist()

def classifySoundkNN(queryFile, targetDir, K):
  
  distances = computeSimilarSounds(queryFile, targetDir)
  print distances
  #note that we go from 1 becasue 0th index will be the query file itself
  classes = (np.array(distances)[1:K+1,2]).tolist()
  
  freqCnt = []
  for ii in range(K):
    freqCnt.append(classes.count(classes[ii]))
  indMax = np.argmax(freqCnt)
  
  print "This sample belongs to class: " + str(classes[indMax])
  
  return classes[indMax]

def clusterSounds(targetDir, nCluster = -1):
  
  dataDetails = fetchDataDetails(targetDir)
  
  ftrArr = []
  infoArr = []
  
  if nCluster ==-1:
    nCluster = len(dataDetails.keys())
  for cname in dataDetails.keys():
    #iterating over sounds
    for sname in dataDetails[cname].keys():
      ftrArr.append(convFtrDict2List(dataDetails[cname][sname]['feature']))
      infoArr.append([sname, cname])
      
  ftrArr = np.array(ftrArr)
  infoArr = np.array(infoArr)
  print ftrArr.shape
  
  ftrArrWhite = whiten(ftrArr)
  
  print nCluster
  centroids, distortion = kmeans(ftrArrWhite, nCluster)
  clusResults = -1*np.ones(ftrArrWhite.shape[0])
  
  for ii in range(ftrArrWhite.shape[0]):
    diff = centroids - ftrArrWhite[ii,:]
    diff = np.sum(np.power(diff,2), axis = 1)
    indMin = np.argmin(diff)
    clusResults[ii] = indMin
    
  for ii in range(nCluster):
    print "Sounds in cluster number " + str(ii+1)
    ind = np.where(clusResults==ii)[0]
    print infoArr[ind]
  
  
    

  
  
  
  
  
      
    
    
    
  