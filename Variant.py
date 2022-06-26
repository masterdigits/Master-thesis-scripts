from genetic_abstract import genetic_abstract,genetic_algorithm
from GPU_engine import GPU_engine
import cudf as df
import cupy as cp

class Variant(genetic_abstract):
    
    def __init__(self,vector: cp.array, GPUEngine: GPU_engine):
        self.vector = vector
        self.GPUEngine = GPUEngine
        super().__init__()
        
    def _perform_mutation(self):
        '''
        Strategia prosta Pętla iteruje po każdym elemencie 
        wektora próbki i z ustalonym prawdopodobieństwem odwraca 
        jego wartość (prawdopodobieństwo powinno być ekstremalnie 
        niskie) 

        Strategia szybka Losowana jest ustalona liczba indeksów, 
        dla których wartości w wektorze należy odwrócić. 

        Strategia naturalna Należy wyznaczyć wzór obliczania 
        prawdopodobieństwa mutacji dla alleli aby każdy z nich 
        miał takie samo prawdopodobieństwo zmutować, ale mutacja 
        każdego allelu zmniejsza prawdopodobieństwo, że kolejne 
        będą mutować. Mutacje są wykonywane dopóki prawdopodobieństwo 
        mutacji nie spadnie do 0. 
        '''
        
        number_of_mutations = int(.001 * len(self.vector) // 2)
        args_zeros = cp.random.choice(cp.argwhere(self.vector>=0).flatten(),number_of_mutations,replace=False)
        args_ones = cp.random.choice(cp.argwhere(self.vector>=0).flatten(),number_of_mutations,replace=False)
        cp.put(self.vector,args_zeros,0)
        cp.put(self.vector,args_ones,1)
       
    def crossover(self,subject: genetic_abstract) -> genetic_abstract:
        '''
        strategia:
        losowany jest wektor zer i jedynek długości oryginalnego
        wektora. Jeżeli pod danym indeksem znajduje się zero to wybierana 
        jest wartość z wektora na self, a jeżeli jedynka to z wektora na 
        subject. Następnie wektor poddawany jest testowi. Jeżeli nie przechodzi 
        testu operacja jest powtarzana.
        '''
        
        vector = cp.zeros_like(self.vector)
        
        args1 = cp.argwhere(self.vector==1).flatten()
        args2 = cp.argwhere(subject.vector==1).flatten()
        
        args = cp.unique(cp.concatenate((args1,args2)))
        length = min(self.vector.sum().item(), len(args))
        args = cp.random.choice(args,length,replace=False)
        
        cp.put(vector,args,1)
        
        return Variant(vector,self.GPUEngine)
    
    def evaluate_fitness(self):
        '''
        dopasowanie wyznaczane jest na podstawie obliczanych miar
        '''
        
        return 1 - self.GPUEngine.f2(self.vector)
