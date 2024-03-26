### The only things you'll have to edit (unless you're porting this script over to a different language) 
### are at the bottom of this file.
import urllib
import email
import email.message
import email.encoders
import sys
import pickle
import json
import base64
import numpy as np
import subprocess
import os
import matplotlib.pyplot as plt
# Python 2 and 3 compatibility:
try: input = raw_input
except: pass
plt.ion()
""""""""""""""""""""
""""""""""""""""""""

class NullDevice:
    def write(self, s):
        pass

def submit():   
    print ('==\n== Submitting Solutions: Programming Assignment-2\n==')
 
    (email, password) = loginPrompt()
    if not email:
        print ('!! Submission Cancelled')
        return
    
    print ('\n== Connecting to Coursera ... ')
    
    # Part Identifier
    (partIdx, sid) = partPrompt()
    
    # submitting all the parts at once
    if partIdx == 'all':
        rep = submitSolution_all_parts(email, password)
        #print rep
        if b'element' in rep:
            print ('\n== Your submission has been accepted and will be graded shortly.')
        else:
            wrongSubmission()     
        
    # error in the process
    elif partIdx == 'error':
        print ('== Submission Cancelled')
        return
    
    # submiting one part
    elif isinstance(partIdx, int):
        if partIdx in range(len(partIds)):
            rep = submitSolution(email, password, output(partIdx), partIdx)
            #print rep
            if b'element' in rep:
                print ('\n== Your submission has been accepted and will be graded shortly.')
            else:
                wrongSubmission()            
        else:
            print ('== Wrong number of part, Submission Cancelled')
            return
    


# =========================== LOGIN HELPERS - NO NEED TO CONFIGURE THIS =======================================

def loginPrompt():
    email = input('Login (Email adress):')
    password = input('Submission Token (from the assignment page. This is NOT your own account\'s password): ')
    return email, password

def partPrompt():
    print ('Hello! These are the assignment parts that you can submit:')
    counter = 0
    for part in partFriendlyNames:
        counter += 1
        print (str(counter) + ') ' + partFriendlyNames[counter - 1])
    print ('\n== Once your parts are correct, you have to submit them all at once by entering "all"')
    partIdx = input('Please enter which part you want to submit (1-' + str(counter) + ' or all): ')   
    try:
        partIdx = int(partIdx)-1
        return (partIdx, partIds[partIdx-1] )
    except (ValueError, IndexError): 
        if partIdx == 'all':
            return ('all','all')
        else:
            return ('error', 'error')

def submit_url():
    """Returns the submission url."""
    return "https://www.coursera.org/api/onDemandProgrammingScriptSubmissions.v1"
    #return "https://class.coursera.org/" + URL + "/assignment/submit"

def get_parts(partIdx, output):
    part = LIST_PARTIDS[partIdx]
    parts = {}
    for parti in LIST_PARTIDS:
        parts[parti] = {}
    output = output + '\n\n\n' + source(partIdx) # concatenating with the source code for log
    #print source(partIdx)
    parts[part] = {"output" : output}
    return parts
    
def submitSolution(email_address, secret, output, partIdx):
    """Submits a solution to the server. Returns (result, string)."""
    if output == '':
        print ('')
        print ("== Submission failed: Please correct and resubmit.")
        sys.exit(1)
    else:
        output_64_msg = email.message.Message()
        output_64_msg.set_payload(output)
        email.encoders.encode_base64(output_64_msg)
    parts = get_parts(partIdx, output_64_msg.get_payload())
    
    values = { "assignmentKey" : ASSIGNMENT_KEY, \
             "submitterEmail" : email_address, \
             "secret" : secret, \
             "parts" : parts, \
           }
    response = send_request_new(values)
    return response

def get_all_parts(outputs):
    parts = {}
    for idx, parti in enumerate(LIST_PARTIDS):
        if outputs[idx]:
            parts[parti] = {"output" : outputs[idx]}
        else:
            parts[parti] = {}
    return parts

def submitSolution_all_parts(email_adress, secret):
    """Submit all parts at once"""
    outputs = []
    for idx in range(len(LIST_PARTIDS)):
        out = output(idx)
        if out == '':
            outputs.append(None)
        else:
            output_64_msg = email.message.Message()
            output_64_msg.set_payload(out)
            email.encoders.encode_base64(output_64_msg)
            outputs.append(output_64_msg.get_payload() + '\n\n\n' + source(idx)) # concatenating the source code for log
    parts = get_all_parts(outputs)
    values = { "assignmentKey" : ASSIGNMENT_KEY, \
             "submitterEmail" : email_adress, \
             "secret" : secret, \
             "parts" : parts, \
           }
    response = send_request_new(values)
    return response
    
# old version that do not support long text
def send_request(values):
    req = 'curl -s -X POST -H "Cache-Control: no-cache" -H "Content-Type: application/json" -d ' +\
    "'" + str(json.dumps(values)) + "' " + \
    "'https://www.coursera.org/api/onDemandProgrammingScriptSubmissions.v1'"
    print ('\n== The request sent to Coursera: \n' + req)
    output = subprocess.check_output(req, shell=True)
    return output

def send_request_new(values):
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }
    url = submit_url()  
    #data = urllib.urlencode(values)
    try:
        import urllib2
        req = urllib2.Request(url, json.dumps(values), headers)
        response = urllib2.urlopen(req)
        string = response.read().strip()
        #print values
        return string
    except Exception as e:
# using requests module
        try:
            import requests
        except ImportError:
            raise Exception('Something got wrong. Try installing Requests module and retry ("pip install requests")')
        r = requests.post(url, headers=headers, data=str(json.dumps(values)))
        #print values, r.content
        return r.content

## This collects the source code (just for logging purposes) 
def source(partIdx):
    # open the file, get all lines
    f = open(sourceFiles[partIdx])
    src = f.read() 
    f.close()
    # This was used for encoding the source code
    source_64_msg = email.message.Message()
    source_64_msg.set_payload(src)
    email.encoders.encode_base64(source_64_msg)
    return source_64_msg.get_payload()

def convertNpObjToStr(obj):
    """
    if input object is a ndarray it will be converted into a dict holding dtype, shape and the data base64 encoded
    """
    if isinstance(obj, np.ndarray):
        obj = np.ascontiguousarray(obj)# this is a very important step, because many times the np object doesn't have its content in continous memory
        data_b64 = base64.b64encode(obj.data)
        return json.dumps(dict(__ndarray__=data_b64.decode("utf-8"),dtype=str(obj.dtype),shape=obj.shape))
    return json.dumps(obj)

def wrongOutputTypeError(outType, part):
    print ("\n Type error in Part " + str(part+1) + " - The output data type of your function doesn't match the expected data type: (" + str(outType) + ").")
    #print "== Submission failed: Please correct and resubmit."

def wrongSubmission():
    print ("\n== Submission failed: Please check your email and token again and resubmit.")

############ BEGIN ASSIGNMENT SPECIFIC CODE - YOU'LL HAVE TO EDIT THIS ##############

from A2Part1 import genSine
from A2Part2 import genComplexSine
from A2Part3 import DFT
from A2Part4 import IDFT
from A2Part5 import genMagSpec

# DEFINE THE ASSIGNMENT KEY HERE
ASSIGNMENT_KEY = 'KvY3iVfNEeagmwqcM_3G5w'

# DEFINE THE PartIds in this list for each PA
LIST_PARTIDS = ['lUa09', 'vb4eQ', '6R3jc', '3fdCd']#, 'UHfAa'] ################## CHANGE THE PART IDS HERE !!

# the "Identifier" you used when creating the part
partIds = ['A2-part-1', 'A2-part-2', 'A2-part-3', 'A2-part-4']#, 'A2-part-5']

# used to generate readable run-time information for students
partFriendlyNames = ['Generate a sinusoid', 'Generate a complex sinusoid', 'Implement the DFT', 'Implement the IDFT']#, 'Compute the magnitude spectrum (optional)'] 
# source files to collect (just for our records)
sourceFiles = ['A2Part1.py', 'A2Part2.py', 'A2Part3.py', 'A2Part4.py']#, 'A2Part5.py']

def output(partIdx):
    """Uses the student code to compute the output for test cases."""
    outputString = ''
    filename = open('testInputA2.pkl','rb')
    try: ## load the dict containing output types and test cases
        dictInput = pickle.load(filename,encoding='latin1')  ## python3
    except TypeError:
        dictInput = pickle.load(filename)  ## python2 
        
    testCases = dictInput['testCases']
    outputType = dictInput['outputType']
  
    if partIdx == 0: # This is A2-part-1: genSine
        for line in testCases['A2-part-1']:
            answer = genSine(**line)
            if outputType['A2-part-1'][0] == type(answer):
                outputString += convertNpObjToStr(answer) + '\n'
            else:
                wrongOutputTypeError(outputType['A2-part-1'][0],partIdx)
                return ''
                #sys.exit(1)
  
    elif partIdx == 1: # This is A2-part-2: genComplexSine
        for line in testCases['A2-part-2']:
            answer = genComplexSine(**line)
            if outputType['A2-part-2'][0] == type(answer):
                outputString += convertNpObjToStr(answer) + '\n'
            else:
                wrongOutputTypeError(outputType['A2-part-2'][0],partIdx)
                return ''
                #sys.exit(1)    
      
    elif partIdx == 2: # This is A2-part-3: DFT
        for line in testCases['A2-part-3']:
            answer = DFT(**line)
            if outputType['A2-part-3'][0] == type(answer):
                outputString += convertNpObjToStr(answer) + '\n'
            else:
                wrongOutputTypeError(outputType['A2-part-3'][0],partIdx)
                return ''
                #sys.exit(1)        
      
    elif partIdx == 3: # This is A2-part-4: IDFT
        for line in testCases['A2-part-4']:
            answer = IDFT(**line)
            if outputType['A2-part-4'][0] == type(answer):
                outputString += convertNpObjToStr(answer) + '\n'
            else:
                wrongOutputTypeError(outputType['A2-part-4'][0],partIdx)
                return ''
                #sys.exit(1)         

    elif partIdx == 4: # This is A2-part-5: genMagSpec
        for line in testCases['A2-part-5']:
            answer = genMagSpec(**line)
            if outputType['A2-part-5'][0] == type(answer):
                outputString += convertNpObjToStr(answer) + '\n'
            else:
                wrongOutputTypeError(outputType['A2-part-5'][0],partIdx)
                return ''
                #sys.exit(1)         

    return outputString.strip()

submit()
