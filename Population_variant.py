from genetic_abstract import genetic_abstract,genetic_algorithm
from Variant import Variant
import numpy as np
import pandas as pd

class Population_variant(genetic_abstract):
    
    def __init__(self,variant: Variant):
        self.variant = variant
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
        
        n = 10 #liczba indeksów do zmiany
        
        vector = self.variant.vector.copy()
        indices = np.random.randint(0,len(vector),10)
        for idx in indices:
            vector[idx] = 1 - vector[idx]
            
        while 50 <= vector.sum() <= 300:
            for idx in indices:
                vector[idx] = 1 - vector[idx]
        
        self.variant = Variant(vector,self.variant.pop1_ids,self.variant.pop2_ids,self.variant.sqlEngine,self.variant.table)
        
    def crossover(self,subject: genetic_abstract) -> genetic_abstract:
        '''
        strategia:
        losowany jest wektor zer i jedynek długości oryginalnego
        wektora. Jeżeli pod danym indeksem znajduje się zero to wybierana 
        jest wartość z wektora na self, a jeżeli jedynka to z wektora na 
        subject. Następnie wektor poddawany jest testowi. Jeżeli nie przechodzi 
        testu operacja jest powtarzana.
        '''
        
        vector = np.empty_like(self.variant.vector)
        indices = np.random.randint(0,1,len(vector))
        while 50 <= vector.sum() <= 300:
            for i in range(len(self.variant.vector)):
                vector[i] = (self.variant.vector[i],subject.variant.vector[i])[indices[i]]
        
        return Population_variant(Variant(vector,
                                     self.variant.pop1_ids,
                                     self.variant.pop2_ids,
                                     self.variant.sqlEngine,
                                     self.variant.table))

    def evaluate_fitness(self):
        '''
        dopasowanie wyznaczane jest na podstawie obliczanych miar
        '''
        #fitness_vector = np.empty_like(weiges)
        
        #self.variant.build_tree()
        
        #fitness_vector[0] = self.variant.f2() * weiges[0]
        #fitness_vector[1] = self.variant.AUC() * weiges[1]
        #fitness_vector[2] = self.variant.kullback_leibler() * weiges[0]
        
        #NET = fitness_vector.sum()
        NET = 1- self.variant.f2()
        return NET