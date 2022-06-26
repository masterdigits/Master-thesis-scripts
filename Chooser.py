from GPU_engine import GPU_engine
import cupy as cp
from variant import variant
from random import choice

class Chooser:
    '''
    Obiekt klasy losuje określoną liczbę zestawów 
    wariantów i dla każdego z nich sprawdza który 
    z jego wariantów jest najbardziej istotny w kontekście 
    całego zbioru. Następnie zapisuje zestaw tych
    wariantów, które są istotne najczęściej.
    '''
    
    def __init__(self,GPUEngine: GPU_engine, C: cp.uint32, length: cp.uint32):
        self.GPUEngine = GPUEngine
        self.C = C
        self.length = length
        
        print('Encoding...')
        self.sets = list()
        self.variants = [variant(pos) for pos in range(self.GPUEngine._f2_array.shape[0])]
        
        print('Generating... ',end='')
        self.generate()
        #print('done\nCounting...')
        #self.countForAll()
        print('\ndone')
        #self.returnVariants()
        
    def generate(self):
        for _ in range(self.C):
            self.sets.append(self.choice_no_return(self.variants,self.length))
            
            
    def choice_no_return(self, array: list, length: cp.uint32):
        if length > len(array):
            raise ValueError(f"You cannot randomly choose {length} elements without return from {len(array)} long array.")
        new_array = []
        while len(new_array) < length:
            choose = choice(array)
            if choose not in new_array:
                new_array.append(choose)
                
        return new_array
    
    def countForAll(self):
        for i in range(self.C):
            variants_set = self.sets[i]
            self.countVariant(variants_set)
            print(f'\r[{"#"*int(100*((i+1)/self.C))}{" "*int(100*(1-(i+1)/self.C))}] {i+1}/{self.C}({100*(i+1)/self.C}%)',end='')
    
    def countVariant(self,variant_set: list):
        position = [variant.position for variant in variant_set]
        vector = cp.zeros(len(self.variants))
        vector.put(position,1)
        f2_ref = self.GPUEngine.f2(vector)
        for pos in position:
            vector[pos] = 0
            f2_new = self.GPUEngine.f2(vector)
            diff = (f2_ref - f2_new).item()
            self.variants[pos].append(diff)
            vector[pos] = 1
            
    def returnVariants(self):
        return self.variants