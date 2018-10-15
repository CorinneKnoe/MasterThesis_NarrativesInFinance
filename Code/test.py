# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 11:40:29 2018

@author: corin
module or script - what is going on?
"""
import os, sys

if __name__ == '__main__':
    
    print('its running')
    #change path to directory of txt data
    this_path = os.path.abspath(os.path.dirname(__file__)) #take absolute path of file, to construct relative path to data
    base_path = os.path.dirname(this_path)
    path = os.path.join(base_path, "FACTIVA_Data")
    #path ="C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/FACTIVA_Data/" #absolute path to the txt files
    print(path) #setting working directory
