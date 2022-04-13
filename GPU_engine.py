from blazingsql import BlazingContext
import pandas as pd
import cudf as cf

class GPU_engine:
    '''
    Obiekt klasy jest odwołaniem do silnika BlazingSQL.
    Klasa definiuje zestaw metod, ktore potrafią uruchamiać
    pożądane polecenia SQL na zbiorze przypisanym do 
    instancji klasy.
    '''
    
    def __init__(self):
        self.engine = BlazingContext()
        self.tables = list()
        
    def addTable(self,name: str,table: pd.DataFrame) -> None:
        if name in self.tables:
            raise ValueError(f'Table named "{name}" already exists')
            
        self.engine.create_table(
            name,
            table
        )
        self.tables.append(name)
        
    def run(self, select: tuple, from_: str, where: dict = None, group_by: tuple = None, having: dict = None, order_by: tuple = None) -> pd.DataFrame:
        '''
        Przyjmuje parametry zapytania w postaci słów kluczowych.
        Paraemtry te stanowią klauzule zapytania SQL.
        '''
        
        query = 'SELECT '
        query += ', '.join(select)
        
        if from_ not in self.tables:
            raise ValueError(f'Table named "{from_}" does not exists.')
        query += ' FROM ' + from_
        
        if where is not None:
            query += ' WHERE '
            query += ', '.join([f'{k}={v}' for k,v in where.items()])
            
        if group_by is not None:
            query += ' GROUP BY '
            query += ', '.join(group_by)
            
        if having is not None:
            query += ' HAVING '
            query += ', '.join([f'{k}={v}' for k,v in having.items()])
        
        if order_by is not None:
            query += ' ORDER BY '
            query += ', '.join(order_by)
            
        #print(query)
            
        return self.engine.sql(query)