# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 10:37:47 2021

@author: Stankevix
"""

# In[0]:
    
import postgres_conn as pgc


# In[1]: Scripts SQL 
    
SCRIPT01 = '''

select * from queimadas_brasil_reservas

'''

# In[2]: Conectar com Postgres SQL

pwd = input("Informe a senha do database: ")
    
conn = pgc.db_conn('localhost','spatial','postgres',pwd)
conn.postgres()

# In[3]:

queimadas = conn.get_dataframe(SCRIPT01)


# In[3]: ensemble

queimadas.info()

# In[3]: ensemble




   
