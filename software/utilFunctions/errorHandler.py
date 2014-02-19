

def printError(errorID):
    
    if errorID == 1:
        print "Error opening file"
        
        
def printWarning(warningID):
    
    if warningID ==1:
        print "\n"
        print "-------------------------------------------------------------------------------"
        print "Warning:"
        print "Cython modules for some of the core functions were not imported."
        print "The processing might be significantly slower in such case"
        print "Please refer to the README file for instructions to compile cython modules"
        print "https://github.com/MTG/sms-tools/blob/master/README.md"
        print "-------------------------------------------------------------------------------"
        print "\n"