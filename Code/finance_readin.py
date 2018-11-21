# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 11:40:29 2018

@author: corin
read in financial data as prepared in csv
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
    
    #set start and end dates, cut df to fit windows we want
    start = datetime.datetime.strptime('01.10.1998', '%d.%m.%Y')
    end = datetime.datetime.strptime('01.10.2018', '%d.%m.%Y')
    financedf = financedf.loc[(financedf.Date >= start) & (financedf.Date < end), :]
    
    
    
   