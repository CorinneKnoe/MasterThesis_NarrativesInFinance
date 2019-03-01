# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 12:12:21 2019

@author: corin
"""
#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
as suggested by my supervisor, Dr. Gruber, I include the most common bigrams in 
the word doc feature matrix
'''
import operator
import time
import sys
import os
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import math
import datetime

if __name__ == "__main__":
        
    #Read in the text dataframe from a csv
    #-------------------------------------
    path ="C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/FACTIVA_Data/" #absolute path to the txt files
    os.chdir(path)

    #create data frame by reading csv
    textdf = pd.read_csv('TokenizedArticles.csv', sep = ',')
    
    #count the most common bigrams in the text data
    BigramCount = {}
    for Line in range(len(textdf)):
        UnigramsInArticle = textdf.loc[Line , 'Article'].split()
        for NoOfUnigram in range(len(UnigramsInArticle)-1):
            bigram = UnigramsInArticle[NoOfUnigram] + ' ' + UnigramsInArticle[NoOfUnigram +1]
            if bigram not in BigramCount.keys():
                BigramCount[bigram] = 0
            BigramCount[bigram] += 1
    
    TopBigrams = sorted(BigramCount.items(), key=operator.itemgetter(1), reverse=True)[:50]
    
    TopBigrams = [Bigram[0] for Bigram in TopBigrams]
    ManualListBadBigrams = ['wsj com', 'fed said', 'year treasury']
    for UnnecessaryBigram in ManualListBadBigrams:
        TopBigrams.remove(UnnecessaryBigram)
        
    #the most common bigrams should be in the text as bigrams, not split in two unigrams
    for Line in range(len(textdf)):
        Article = []
        UnigramsInArticle = textdf.loc[Line , 'Article'].split()
        Skip = False
        for NoOfUnigram in range(len(UnigramsInArticle)-1):
            if not Skip: 
                bigram = UnigramsInArticle[NoOfUnigram] + ' ' + UnigramsInArticle[NoOfUnigram +1]
                if bigram in TopBigrams:
                    Article.append(bigram)
                    Skip = True
                else:
                    Article.append(UnigramsInArticle[NoOfUnigram])
            else:
                Skip = False #only skip once
        #add last word in Article if it was not already part of the last bigram
        if len(Article[-1].split()) == 1: #we know last word was not added in a bigram
            Article.append(UnigramsInArticle[-1])            
        #save new bigram rich list in df
        textdf.loc[Line , 'Article'] = Article
        
    #save new tokenized text data with bigrams to data frame
    
    