# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 10:37:47 2021

@author: Stankevix
"""


# In[0]:
    
import postgres_conn as pgc


# In[1]: Scripts SQL 
    
SCRIPT01 = '''
select * from brasil LIMIT 10
'''

# In[2]: Conectar com Postgres SQL

pwd = input("Informe a senha do database: ")
    
conn = pgc.db_conn('localhost','spatial','postgres',pwd)
conn.postgres()

# In[3]:

df_brasil = conn.get_dataframe(SCRIPT01)    
