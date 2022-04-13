import numpy as np
import pandas as pd
import cudf as cf
from GPU_engine import GPU_engine

class Variant:
    
    def __init__(self,vector: np.array, population1_ids: np.array, population2_ids: np.array, sqlEngine: GPU_engine, table_name: str):
        self.vector = vector               # wektor o długości równej liczbie dostępnych wariantów genetycznych. Każda 1 reprezentuje allel, który jest brany pod uwagę w bieżących obliczeniach
        self.pop1_ids = population1_ids    # wektor identyfikatorów (nazw kolumn w tabeli) populacji 1
        self.pop2_ids = population2_ids    # wektor identyfikatorów (nazw kolumn w tabeli) populacji 2
        self.sqlEngine = sqlEngine         # handler do silnika SQL
        self.table = table_name            # nazwa tabeli, w której znajdują się dane
        
        # wyodrębnienie identyfikatorów wariantów genetycznych
        self.indices = self.sqlEngine.run(select=('POS',),from_=self.table)
        self.indices = ['\'' + str(idx) + '\'' for idx, variant in zip(self.indices['POS'].to_arrow().to_pylist(),self.vector) if variant == 1]
        
    def f2(self):
        '''
        Oblicza miarę f2 między populacjami poprzez wyekstrahowanie
        ze zbioru tych wariantów genetycznych, dla których self.vector
        przyjmuje wartość 1 oraz z kolumn przypisanych do danych populacji.
        '''
        idx = [f'POS={ids}' for ids in self.indices]
        where_val = {"1":f"1 AND ({' OR '.join(idx)})"}
        
        pop1 = self.sqlEngine.run(select=self.pop1_ids,from_=self.table,where=where_val)
        pop2 = self.sqlEngine.run(select=self.pop2_ids,from_=self.table,where=where_val)
        tab = cf.DataFrame()
        tab['pop1'] = pop1.sum(axis=1)/(len(self.pop1_ids)*2)
        tab['pop2'] = pop2.sum(axis=1)/(len(self.pop2_ids)*2)
        tab['values'] = (tab['pop1'] - tab['pop2'])**2
        f2_value = tab['values'].sum()
        
        return f2_value
    
    def f3(self):
        pass
    
    def f4(self):
        pass
    
    def AUC(self):
        '''
        Oblicza wartość AUC (Area Under Curve) wyznaczonego klasyfikatora.
        '''
        pass
    
    def kullback_leibler(self):
        '''
        Oblicza wartość dywergencji Kullbacka-Leiblera dla obu populacji.
        W pierwszej kolejności oblicza prawdopodobieństwa p_i dla pierwszej 
        populacji oraz q_i dla drugiej. Nastepnie oblicza wartość dywergencji
        zgodnie z wzorem $\sum_i{p_i log_2{\frac{p_i}{q_i}}}$.
        '''
        select_val = (f'SUM(SQUARE({"+".join(population1_ids)}/{len(population1_ids)}*LOG(({"+".join(population2_ids)}/{len(population2_ids)})/({"+".join(population1_ids)}/{len(population1_ids)}),2))) AS DKL',)
        where_val = {"1":f"1 AND POS IN ({', '.join(self.indices)})"}
        result = self.sqlEngine.run(select=select_val,from_='populacja',where=where_val)
        
        return result.loc[0,'DKL']
    
    def build_tree(self):
        '''
        Służy do zbudowania drzewa. Posługuje się algorytmem indukcji drzew 
        decyzyjnych. Uruchamiany jest na karcie graficznej.
        '''
        pass