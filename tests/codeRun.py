import sys, os, time

sys.path.append(os.path.realpath('../code/spectralModels/'))
sys.path.append(os.path.realpath('../code/spectralTransformations/'))

print "This script tests if all the modules are importable and they atleast run. This script doesn't check how good is the output"

############################################################################################
### First level of test is to check if we are able to even import the module into python ###
############################################################################################

print "Check 1: Importing every module....."

try:
  import harmonicModel
  print "[Success] harmonicModel imported successfully....."
except ImportError:
  print "[Error] problem while importing harmonicModel"
 
try:
  import hprModel
  print "[Success] hprModel imported successfully....."
except ImportError:
  print "[Error] problem while importing hprModel"
 
try:
  import hpsModel
  print "[Success] hpsModel imported successfully....."
except ImportError:
  print "[Error] problem while importing hpsModel"
 
try:
  import sineModel
  print "[Success] sineModel imported successfully....."
except ImportError:
  print "[Error] problem while importing sineModel"
 
try:
  import spsModel
  print "[Success] sps imported successfully....."
except ImportError:
  print "[Error] problem while importing sps"
 
try:
  import stft
  print "[Success] stft imported successfully....."
except ImportError:
  print "[Error] problem while importing stft"
 
try:
  import stftPeaks
  print "[Success] stftPeaks imported successfully....."
except ImportError:
  print "[Error] problem while importing stftPeaks"
 
try:
  import stochasticModel
  print "[Success] stochasticModel imported successfully....."
except ImportError:
  print "[Error] problem while importing stochasticModel"
 
try:
  import hpsModelParams
  print "[Success] hpsModelParams imported successfully....."
except ImportError:
  print "[Error] problem while importing hpsModelParams"
 
try:
  import hpsMorph
  print "[Success] hpsMorph imported successfully....."
except ImportError:
  print "[Error] problem while importing hpsMorph"

try:
  import spsTimeScale
  print "[Success] harmonicModel imported successfully....."
except ImportError:
  print "[Error] problem while importing spsTimeScale"


########################################################################################################
### Second level of test is to check if we are able to atleast run all the modules without crashing  ###
########################################################################################################

print "Check 2: Trying to run defaultTest function of each module......"



try:
  harmonicModel.defaultTest()
  print "[Success] harmonicModel ran successfully....."
except ImportError:
  print "[Error] problem while running defaultTest function in harmonicModel"
 
try:
  hprModel.defaultTest()
  print "[Success] hprModel ran successfully....."
except ImportError:
  print "[Error] problem while running defaultTest function in hprModel"
 
try:
  hpsModel.defaultTest()
  print "[Success] hps ran successfully....."
except ImportError:
  print "[Error] problem while running defaultTest function in hps"

try:
  sineModel.defaultTest()
  print "[Success] sineModel ran successfully....."
except ImportError:
  print "[Error] problem while running defaultTest function in sineModel"
 
try:
  spsModel.defaultTest()
  print "[Success] sps ran successfully....."
except ImportError:
  print "[Error] problem while running defaultTest function in sps"
 
try:
  stft.defaultTest()
  print "[Success] stft ran successfully....."
except ImportError:
  print "[Error] problem while running defaultTest function in stft"
 
try:
  stftPeaks.defaultTest()
  print "[Success] stftPeaks ran successfully....."
except ImportError:
  print "[Error] problem while running defaultTest function in stftPeaks"
 
try:
  stochasticModel.defaultTest()
  print "[Success] stochasticModel ran successfully....."
except ImportError:
  print "[Error] problem while running defaultTest function in stochasticModel"
 
try:
  hpsMorph.defaultTest()
  print "[Success] hpsMorph ran successfully....."
except ImportError:
  print "[Error] problem while running defaultTest function in hpsMorph"

try:
  spsTimeScale.defaultTest()
  print "[Success] spsTimeScale ran successfully....."
except ImportError:
  print "[Error] problem while running defaultTest function in spsTimeScale"


print "Testing completed..."