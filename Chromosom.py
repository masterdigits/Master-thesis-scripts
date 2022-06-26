import numpy as np
import cupy as cp
from Variant import Variant
from GPU_engine import GPU_engine
from genetic_abstract import genetic_abstract, genetic_algorithm, elite_selection_model
from Chooser import Chooser
from variant import variant

class Chromosom:
    def __init__(
        self, 
        name: str,
        tables: tuple, 
        maf_goodness: tuple, 
        chooser_groups_number: int, 
        chooser_length: int,
        E: int,
        E_learn: int,
        N: int,
        variants_number: int,
        debug:bool = False
    ):
        self.chromosom_name = name
        self.epoch_max = E
        self.epoch_since_last_change_max = E_learn
        self.population_len = N
        self.variants_number = variants_number
        self.debug = debug
        
        self.process_length = None
        self.best_one = None
        
        self.gpu = self.create_gpu_engine(tables,maf_goodness)
        self.chooser = self.create_chooser(chooser_groups_number, chooser_length)
        self.GA = self.create_genetic()
        
        
    def create_gpu_engine(self, files: tuple, mafs: tuple) -> GPU_engine:
        file1, file2 = files
        return GPU_engine(file1, file2, mafft_goodness = mafs)
    
    def create_chooser(self, groups_number: int, length: int) -> Chooser:
        return Chooser(self.gpu, groups_number, length)
    
    def create_genetic(self) -> genetic_algorithm:
        return genetic_algorithm(
            self.first_generation_function, 
            elite_selection_model, 
            self.stop_condition
        )
    
    def run(self):
        full_length = self.gpu._f2_array.shape[0] #total SNP`s
        if self.debug:
            print(f'::full_length={full_length}::')
        ## chooser optimizer
        self.chooser.countForAll()
        variants = self.chooser.returnVariants()
        average = np.array([v.sum() for v in variants if len(v.differences) > 0]).mean()
        std_deviation = np.array([v.sum() for v in variants if len(v.differences) > 0]).std()
        standardized = (np.array([v.sum() for v in variants if len(v.differences) > 0])-average)/std_deviation
        a = cp.zeros(full_length)
        l = 0
        args = cp.array(np.argwhere(standardized > l))
        while args.shape[0] > 15000:
            l += .1
            args = cp.array(np.argwhere(standardized > l))
        a.put(args, 1)
        self.process_length = int(a.sum()) #SNP`s to process number
        if self.debug:
            print(f'::l={l}::')
            print(f'::process_length={self.process_length}::')
        print('Optimizer stats:')
        print(f'\tAverage SNP`s coverage: {np.array([len(v.differences) for v in variants]).mean()}')
        print(f'\tTotal number of SNP`s: {full_length}')
        print(f'\tNumber of SNP`s to process: {self.process_length}')
        print(f'\tReference F2 value: {self.gpu.f2(a)}')
        
        ## Genetic Algorithm
        self.gpu._array_filter(a)
        
        print('Running genetic algorithm...')

        self.best_one = self.GA.run()
        if self.debug:
            print(f'\nBest one length: {self.best_one.vector.shape[0]}')
            
        print('Done.')
        
        
    def get_args(self) -> cp.array:
        args = cp.argwhere(self.best_one.vector == 1).flatten()
        return args
        
    
    def __first_generation__(self, sample_length: int):
        ones_prob = self.variants_number/sample_length/2
        rand = cp.random.binomial(
            2,
            ones_prob,
            size=sample_length,
            dtype=cp.uint8
        )

        return Variant(rand, self.gpu)
    
    def stop_condition(self, best_per_population: list) -> bool:
        epoch = len(best_per_population)
        curr_fitness = best_per_population[-1].fitness

        ## jeżeli ostatnie bieżące dopasowanie jest mniejsze niż 0.1
        if curr_fitness < .1:
            return True

        ## jeżeli liczba epok przekroczyła wartość krytyczną
        if epoch >= self.epoch_max:
            return True

        ## jeżeli liczba epok od ostatniej zmiany przekroczyła 50
        last_change = sum([1 for best in best_per_population if best.fitness == curr_fitness])
        if last_change > self.epoch_since_last_change_max:
            return True

        return False
    
    def first_generation_function(self) -> list:
        return [self.__first_generation__(self.variants_number) for _ in range(self.population_len)]