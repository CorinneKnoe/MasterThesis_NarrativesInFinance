# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 11:40:29 2018

@author: corin
create graphs that show the timeseries data over time
"""

import time
import datetime
#nltk.download()
import pandas as pd
import os, sys
import numpy as np

if __name__ == '__main__':
    
    path ="C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/MA_FinancialData/" #absolute path to the txt files
    os.chdir(path)
    
    financedf = pd.read_csv('financeData.csv', sep = ',', float_precision='round_trip') #round trip ensures that no weird decimals appear
    for i in range(len(financedf)): #replace date string with date format
        financedf.iloc[i,0] = datetime.datetime.strptime(financedf.iloc[i,0], '%Y-%m-%d')
    
    financedf.iloc[1,0] == treasurydf.iloc[1,0]
    financedf.iloc[:,1] == treasurydf.iloc[:,1]
    financedf.iloc[5215,11] == treasurydf.iloc[5215,11]
    type(financedf.iloc[0,-1])

    
    print('its running')
    #change path to directory of txt data
    this_path = os.path.abspath(os.path.dirname(__file__)) #take absolute path of file, to construct relative path to data
    base_path = os.path.dirname(this_path)
    path = os.path.join(base_path, "FACTIVA_Data")
    #path ="C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/FACTIVA_Data/" #absolute path to the txt files
    print(path) #setting working directory
