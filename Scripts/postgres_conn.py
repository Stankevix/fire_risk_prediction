# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 20:18:22 2021

@author: Stankevix
"""

# In[0]:
import psycopg2 as pg
import pandas as pd

# In[2]:
class db_conn:
    """ db_conn class for representing and manipulating postgres connection. """

    def __init__(self, server, database, username, pwd):
        self.server = server
        self.database = database
        self.username = username
        self.password = pwd

    def get_server(self):
        return self.server

    def get_database(self):
        return self.database
    def get_username(self):
        return self.username
    
    def postgres(self):
        conn = pg.connect(dbname = self.database, 
                          user=self.username, 
                          password=self.password, 
                          host=self.server)
        self.conn = conn
        return conn
    
    def get_dataframe(self, sql):
        df = pd.read_sql(sql, self.conn)
        return df
        