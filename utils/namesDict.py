import json 

def idx_classes(filepath):
    '''
    INPUT
    - location of the json file
    OUTPUT 
    - Return dictionary
    ACTIONS
    - open jon file 
    - returns the dictionary that has the idx to classes names
    '''
    with open(filepath, "r") as inputfile:
    	classesDict= json.load(inputfile)
    
    return classesDict
