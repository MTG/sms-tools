import os, sys
import freesound as fs
import json

# obtain the API key from freesound.org and add it here
Key = "????????????"

descriptors = [ 'lowlevel.spectral_centroid.mean',
                'lowlevel.spectral_centroid.var',
                'lowlevel.mfcc.mean',
                'lowlevel.mfcc.var',
                'lowlevel.pitch_salience.mean',
                'lowlevel.pitch_salience.var',
                'sfx.logattacktime.mean']
stats = ['mean', 'var']


def downloadSoundsFreesound(queryText = "", API_Key = "", outputDir = "", topNResults = 5, tag=None, duration=None, featureExt = '.json'):
  """
  This function downloads sounds and their descriptors from freesound based on the queryText and the tag specified in the
  input. Additionally to filter the sounds based on the duration you can also specify the duration range.
  
  Inputs:
        queryText (string): query text for the sounds (eg. "violin", "trumpet", "bass", "Carnatic" etc.)
        tag* (string): tag to be used while searching for sounds. (eg. "multisample" etc.)
        duration* (tuple): min and the max duration (seconds) of the sound to filter (eg (1,15)) 
        API_Key (string): your api key, which you can obtain from : www.freesound.org/apiv2/apply/
        outputDir (string): path to the directory where you want to store the sounds and their descriptors
        topNResults (integer): number of results/sounds that you want to download 
  output:
        This function downloads sounds and descriptors and stores them in appropriate folders within outputDir. 
        The name of the directory for each sound is the freesound id of that sound.
        
  NOTE: input parameters with * are optional.
  """ 
  
  #checking if the compulsory input parameters are provided
  if queryText == "":
    print ("\n")
    print ("Provide a query text to search for sounds")
    return -1
    
  if API_Key == "":
    print ("\n")
    print ("You need a valid freesound API key to be able to download sounds.")
    print ("Please apply for one here: www.freesound.org/apiv2/apply/")
    print ("\n")
    return -1
    
  if outputDir == "" or not os.path.exists(outputDir):
    print ("\n")
    print ("Please provide a valid output directory")
    return -1    
  
  #checking authentication stuff
  fsClnt = fs.FreesoundClient()
  fsClnt.set_token(API_Key,"token")  
  
  #creating a filter string which freesound API understands
  if duration and type(duration) == tuple:
    flt_dur = " duration:[" + str(duration[0])+ " TO " +str(duration[1]) + "]"
  else:
    flt_dur = ""
 
  if tag and type(tag) == str:
    flt_tag = "tag:"+tag
  else:
    flt_tag = ""

  #querying freesund
  page_size = 20
  if not flt_tag + flt_dur == "":
    qRes = fsClnt.text_search(query=queryText ,filter = flt_tag + flt_dur,sort="rating_desc",fields="id,name,previews,username,url,analysis", descriptors=','.join(descriptors), page_size=page_size, normalized=1)
  else:
    qRes = fsClnt.text_search(query=queryText ,sort="rating_desc",fields="id,name,previews,username,url,analysis", descriptors=','.join(descriptors), page_size=page_size, normalized=1)
  
  outDir2 = os.path.join(outputDir, queryText)
  if os.path.exists(outDir2):
      os.system("rm -r " + outDir2)
  os.mkdir(outDir2)

  pageNo = 1
  sndCnt = 0
  indCnt = 0
  totalSnds = qRes.count
  #creating directories to store output and downloading sounds and their descriptors
  while(1):
    sound = qRes[indCnt - ((pageNo-1)*page_size)]
    outDir1 = os.path.join(outputDir, queryText, str(sound.id))
    if os.path.exists(outDir1):
      os.system("rm -r " + outDir1)
    os.system("mkdir " + outDir1)
    
    mp3Path = os.path.join(outDir1,  str(sound.previews.preview_lq_mp3.split("/")[-1]))
    ftrPath = mp3Path.replace('.mp3', featureExt)
    
    try:
      fs.FSRequest.retrieve(sound.previews.preview_lq_mp3, fsClnt, mp3Path)
      #initialize dictionary to store features/descriptors
      features = {}
      #obtaining all the features/descriptors
      for desc in descriptors:
        features[desc]=[]
        features[desc].append(eval("sound.analysis."+desc))
      
      #once we have all the descriptors, lets store them in a json file
      json.dump(features, open(ftrPath,'w'))
      sndCnt+=1
      
    except:
      if os.path.exists(outDir1):
        os.system("rm -r " + outDir1)
    
    indCnt +=1
    
    if indCnt%page_size==0:
      qRes = qRes.next_page()
      pageNo+=1
      
    if sndCnt>=topNResults or indCnt >= totalSnds:
      break
    

######
  
downloadSoundsFreesound(queryText = 'trumpet', API_Key = Key, tag = 'single-note',  duration=(0.5, 4), topNResults = 20, outputDir = 'freesound-sounds')
downloadSoundsFreesound(queryText = 'violin', API_Key = Key, tag = 'single-note',  duration=(0.5, 4), topNResults = 20, outputDir = 'freesound-sounds') 
downloadSoundsFreesound(queryText = 'flute', API_Key = Key, tag = 'single-note',  duration=(0.5, 4), topNResults = 20, outputDir = 'freesound-sounds') 
  
