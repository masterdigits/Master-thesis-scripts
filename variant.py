import numpy as np
import cupy as cp

class variant:
    def __init__(self, position: cp.uint32):
        self.position = position
        self.differences = list()
        
    def append(self,value):
        self.differences.append(value)
    
    def average(self):
        diffs = np.array(self.differences)
        return diffs.mean()
    
    def std_dev(self):
        diffs = np.array(self.differences)
        return diffs.std()
    
    def sum(self):
        diffs = np.array(self.differences)
        return diffs.sum()