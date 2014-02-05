import sys, os, time

sys.path.append(os.path.realpath('../models/'))
sys.path.append(os.path.realpath('../transformations/'))

print "This script tests if all the modules are importable and they atleast run. This script doesn't check how good is the output"

############################################################################################
### First level of test is to check if we are able to even import the module into python ###
############################################################################################

print "Check 1: Importing every module....."

try:
  import stft
  print "[Success] stft imported successfully....."
except ImportError:
  print "[Error] problem while importing stft"

try:
  import stftPeaksModel
  print "[Success] stftPeaksModel imported successfully....."
except ImportError:
  print "[Error] problem while importing stftPeaksModel"

try:
  import sineModel
  print "[Success] sineModel imported successfully....."
except ImportError:
  print "[Error] problem while importing sineModel"


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
  import sprModel
  print "[Success] sprModel imported successfully....."
except ImportError:
  print "[Error] problem while importing sprModel"

try:
  import stochasticModel
  print "[Success] stochasticModel imported successfully....."
except ImportError:
  print "[Error] problem while importing stochasticModel"

try:
  import hpsModel
  print "[Success] hpsModel imported successfully....."
except ImportError:
  print "[Error] problem while importing hpsModel"
 
try:
  import spsModel
  print "[Success] spsModel imported successfully....."
except ImportError:
  print "[Error] problem while importing spsModel"

try:
  import sineModel
  print "[Success] sineModel imported successfully....."
except ImportError:
  print "[Error] problem while importing sineModel"
 
try:
  import spsModel
  print "[Success] spsModel imported successfully....."
except ImportError:
  print "[Error] problem while importing spsModel"

try:
  import hpsAnal
  print "[Success] hpsAnal imported successfully....."
except ImportError:
  print "[Error] problem while importing hpsAnal"
 
try:
  import hpsTimeScale
  print "[Success] hpsTimeScale imported successfully....."
except ImportError:
  print "[Error] problem while importing hpsTimeScale"


########################################################################################################
### Second level of test is to check if we are able to atleast run all the modules without crashing  ###
########################################################################################################

print "Check 2: Trying to run defaultTest function of each module......"

try:
  stft.defaultTest()
  print "[Success] stft ran successfully....."
except ImportError:
  print "[Error] problem while running defaultTest function in stft"
 
try:
  stftPeaksModel.defaultTest()
  print "[Success] stftPeaks ran successfully....."
except ImportError:
  print "[Error] problem while running defaultTest function in stftPeaks"
 
try:
  sineModel.defaultTest()
  print "[Success] sineModel ran successfully....."
except ImportError:
  print "[Error] problem while running defaultTest function in sineModel"
 
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
  sprModel.defaultTest()
  print "[Success] sprModel ran successfully....."
except ImportError:
  print "[Error] problem while running defaultTest function in sprModel"

try:
  stochasticModel.defaultTest()
  print "[Success] stochasticModel ran successfully....."
except ImportError:
  print "[Error] problem while running defaultTest function in stochasticModel"

try:
  hpsModel.defaultTest()
  print "[Success] hpsModel ran successfully....."
except ImportError:
  print "[Error] problem while running defaultTest function in hpsModel"

try:
  spsModel.defaultTest()
  print "[Success] spsModel ran successfully....."
except ImportError:
  print "[Error] problem while running defaultTest function in spsModel"
  
try:
  hpsTimeScale.defaultTest()
  print "[Success] hpsTimeScale ran successfully....."
except ImportError:
  print "[Error] problem while running defaultTest function in hpsTimeScale"


print "Testing completed..."