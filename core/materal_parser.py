

class MaterialParser(object):
    '''
    classdocs
    '''


    def __init__(self, file_name):
        '''
        Constructor
        '''
        self.file_name = file_name
    
    def parse(self):
        contents = [] 
        mtl = None
        for line in open(self.file_name, "r"):
            line = line.split('#')[0].strip()
            if not line:
                continue
            values = line.split()
            if not values: 
                continue
            if values[0] == 'newmtl':
                mtl = {'map' : [0, 0, 0]}
                contents.append((values[1], mtl))
            elif mtl is None:
                raise ValueError, "mtl file doesn't start with newmtl stmt"
            else:
                mtl[values[0]] = map(float, values[1:])
        return contents 
            
    
