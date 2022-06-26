import pandas as pd
import cudf as df
import cupy as cp
from datetime import datetime

class GPU_engine:
    '''
    Obiekt klasy zawiera referencję do tabel
    zawierających populacje. 
    Klasa definiuje zestaw metod pozwalających 
    manipulować tabelami w taki sposób aby wydobywać 
    podtabele umożliwiające obliczenia na GPU.
    '''
    
    def __init__(self, *tables,mafft_goodness: tuple):
        self.tables = []
        self.start = datetime.now()
        for tab in tables:
            t = pd.read_csv(tab,sep='\t',dtype=cp.uint8,compression='gzip',index_col='POS')
            print(f'Wczytano {tab}: {datetime.now()}({datetime.now()-self.start})')
            self.tables.append(cp.array(t.sum(axis=1)/(t.shape[1]*2),dtype=cp.float64))
            print(f'Przeliczono {tab}: {datetime.now()}({datetime.now()-self.start})')
        self._f2_array = None
        #self._f3_array = None
        #self._f4_array = None
        #self._dkl_array = None
        self._mafft = None
        
        self._mafft_counter_()
        print(f'Zliczono: {datetime.now()}({datetime.now()-self.start})')
        self._mafft_filter_(mafft_goodness[0],mafft_goodness[1])
    
    def _mafft_counter_(self):
        if len(self.tables) == 2:
            self._f2_array = (self.tables[0]-self.tables[1])**2
            self._mafft = self._f2_array/(self.tables[0]+self.tables[1])**2
        else:
            raise ValueError('Something went wrong.')
        
    def _mafft_filter_(self,mafft_goodness_min: cp.float32,mafft_goodness_max: cp.float32):
        indices = cp.argwhere(cp.logical_and(self._mafft >= mafft_goodness_min,self._mafft <= mafft_goodness_max)).flatten()
        print(f'Wyznaczono indeksy: {datetime.now()}({datetime.now()-self.start})')
        self._f2_array = self._f2_array[indices].flatten()
        print(f'Wybrano {self._f2_array.shape[0]} pozycji: {datetime.now()}({datetime.now()-self.start})')
        
    def _array_filter(self, vector: cp.array):
        if self._f2_array is None:
            raise ValueError('f2 array does not exist. Check imported data.')
            
        self._f2_array = self._f2_array[cp.argwhere(vector==1)].flatten()
        
    def f2(self,vector:cp.array) -> cp.float64:
        if self._f2_array is None:
            raise ValueError('f2 array does not exist. Check imported data.')
            
        return self._f2_array[cp.argwhere(vector==1)].sum()/vector.sum()
        
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
        pass
    
    def build_tree(self):
        '''
        Służy do zbudowania drzewa. Posługuje się algorytmem indukcji drzew 
        decyzyjnych. Uruchamiany jest na karcie graficznej.
        '''
        pass