import pickle

PA = 'A3'

def load(partId, caseId=1):
	"""
	This function returns the example test-cases for a specific part of an assignment.
	Input:
		partId (int) = part number of the assignment (1 for A*Part1, 2 for A*Part2 and so on)
		caseId (int) = caseId = k to return the kth test case. Typically there are two per part.
	Output:
		testcase (dict) = {'input': <input test case>, 'output': <expected output for the input test case>}
	"""
	try:
		data = pickle.load(open('testInput%s.pkl'%PA,'rb'), encoding='latin1')  ## python3
	except TypeError:
		data = pickle.load(open('testInput%s.pkl'%PA,'rb'))  ## python2
		
	part = u'%s-part-%d'%(PA, partId)
	if part not in data['exampleInputs']:
		print ("Please provide a valid partId (>=1), number of parts in this assignment are %d"%(len(data['exampleInputs'].keys())))
		return {'input':None, "output":None}
	if caseId > len(data['exampleInputs'][part]) or caseId <=0:
		print ("Please provide a valid caseId (>=1), number of test cases in this assignment are %d"%(len(data['exampleInputs'][part])))
		return {'input':None, "output":None}

	return {'input': data['exampleInputs'][part][caseId-1], 'output': data['exampleOutputs'][part][caseId-1]}



