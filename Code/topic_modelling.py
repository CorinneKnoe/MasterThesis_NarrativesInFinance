# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 15:33:50 2018

@author: corin
"""

import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import time
import datetime
import pandas as pd
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import pylab
import seaborn; seaborn.set()
import re

from gensim import models, corpora
from nltk import word_tokenize
#from nltk.tokenize import word_tokenize
# =============================================================================
# from .prep_preprocessing import tokenize
# from .prep_processing import removestopwords
# from .prep_processing import stemmer_porter
# from .prep_processing import albhabetizer
# =============================================================================
import pyLDAvis.sklearn


#import re
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfTransformer

#import nltk
#nltk.download('stopwords')

#from sklearn.decomposition import LatentDirichletAllocation 
from nltk.corpus import gutenberg
#nltk.download()
#from nltk.book import *
from wordcloud import WordCloud
from nltk.probability import ConditionalFreqDist
from nltk import FreqDist
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

#from nltk.draw import dispersion_plot
#nltk.download('punkt')

def tokenize(text):
    '''apply the nltk tokenizer and lower case everything, gives out a 
    list of tokens'''
    text = nltk.word_tokenize(text.strip()) #tokenize the text of the article    
    #text = [w.lower() for w in text]
    return text

def removestopwords(text):
    '''remove the stop words of a list of tokens'''
    stop = stopwords.words('english')
    text = [w.lower() for w in text if w.lower() not in stop]
    text = " ".join(text) #to remove all unnecessary white spaces
    return text.strip()

def stemmer_porter(text):
    '''Porter stemmer - split text and convert all words back to 
    their stem, e.g. running -> run, return a text of the stemmed words'''
    porter = PorterStemmer()
    stem = [porter.stem(word) for word in text.split()]
    return " ".join(stem).strip()

def albhabetizer(text):
    '''removing all punctuation, non-letter characters and white spaces'''
    text.strip()
    text = (re.sub('[\W]+', ' ', text))      #remove non-word characters and make text lowercase
    text = (re.sub('[\d]+', '', text)) #to remove numbers [0-9]
    " ".join(text.split()) #to remove all unnecessary white spaces
    return text.strip()







if __name__ == '__main__':
    
    #Read in the text dataframe from a csv
    #-------------------------------------
    path ="C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/FACTIVA_Data/" #absolute path to the txt files
    os.chdir(path)
    os.getcwd()
    #create data frame by reading csv
    textdf = pd.read_csv('FACTIVArticles.csv', sep = ',')
    #turn first colum from string to datetime
    textdf['Date'] = pd.to_datetime(textdf['Date'])
     
    # prepare data and filter out stopwords
    textdf['Article'] = textdf['Article'].apply(tokenize)
    textdf['Article'] = textdf['Article'].apply(removestopwords)
    textdf['Article'] = textdf['Article'].apply(albhabetizer)
    textdf['Article'] = textdf['Article'].apply(stemmer_porter)
    
    
    # Read in the FEd meetings dates
    #---------------------------------
    path ="C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/MA_FinancialData/FED_Data" #absolute path to the txt files
    os.chdir(path) #setting working directory
    
    #read in the FED data on target rate
    adjustdf = pd.read_csv('adjustments.csv', sep = ',')
    #drop rows if empty
    while math.isnan(adjustdf.iloc[len(adjustdf)-1,1]):
        adjustdf = adjustdf[:-1] #drop last row if it is empty
        
    for i in range(len(adjustdf)): #replace date string with date format
        adjustdf.iloc[i,0] = datetime.datetime.strptime(adjustdf.iloc[i,0], '%d/%m/%Y')
    
    #rearrange dataframes to ascending order
    #adjustdf.sort_values('Date', inplace = True)
    
    start = datetime.datetime.strptime('01.10.1998', '%d.%m.%Y')
    end = datetime.datetime.strptime('01.10.2018', '%d.%m.%Y')
    feddf = adjustdf.loc[(adjustdf.Date >= start) & (adjustdf.Date < end), :]
    
    
    #Gensim implementation
    #----------------------
    
    l = []
    for i in range(len(textdf['Article'])):
        l.append('xe' in textdf.loc[i, 'Article'])
    any(l)
    l.count(True)
    stop = stopwords.words('english')
    
    # For gensim we need to tokenize the data and filter out stopwords
    
    
    #Sklearn application
    #---------------------
    
    # prepare data and filter out stopwords
    
    #Transformation Into Feature Vectors that count how often words appear in different speeches
    count = CountVectorizer(max_df = 0.5)
    #count = TfidfVectorizer(max_df = 0.8)
    X = count.fit_transform(textdf['Article'].values)  #sparse matrix
    #fit LDA 
    lda = LatentDirichletAllocation(n_components=2, random_state=123, learning_method='batch')
    X_topics = lda.fit_transform(X)    
      
        
    #print 5 most important words of every topic
    print("The five most important words of the five topics are:")
    print()
    n_top_words = 10
    feature_names = count.get_feature_names()
    for topic_idx, topic in enumerate(lda.components_):
        print("Topic %d:" % (topic_idx + 1))
        print(" ".join([feature_names[i]
        for i in topic.argsort()\
            [:-n_top_words - 1:-1]]))
    
    
    # Stacked Bar plot
    #------------------
    #list of all meeting dates
    meetinglist = []
    for d in range(len(feddf['Date'])):
        meetinglist.append((str(feddf.iloc[d, 0])[:10]))
        
        
    weights=[]  #take the average of topic porbabilities over all articles per meeting date 
    for day in list(feddf['Date']):
        dayup = day + datetime.timedelta(days=2) #look for articles day after, needs day = 2 to cover sae and following day, 00:00 is start of day
        daylow = day - datetime.timedelta(days=1) #look for articles day after, and everything between
        indexlist = list(textdf.loc[(textdf.Date >= daylow) & (textdf.Date <= dayup), :].index)
        line=[]
        for i in indexlist:
            line.append(list(X_topics[i]))
        weights.append(list(np.mean(line, axis=0)))
    
    #prepare plot - inspiration and code examples from https://de.dariah.eu/tatom/topic_model_visualization.html
    N, K = len(weights), len(weights[0]) #N is numer of meeting dates, K is numer of topics
    ind = np.arange(N)
    width = 0.5 
    plots = []
    height_cumulative = np.zeros(N)
    
    s = []
    for x in range(K):
        t = [] # alist with values for all meetings for one topic
        for entry in weights:
            t.append(entry[x])
        s.append(t)  #a list with K list, each with the weights on the respective topic per meetgin date
        
    fig = plt.figure(figsize=(14,4))
    height_cumulative = []
    for k in range(K):
        color = seaborn.color_palette('deep')[k]
        if k == 0:
            p = plt.bar(ind, s[k], width, color=color)
        else:
            p = plt.bar(ind, s[k], width, bottom=height_cumulative, color=color)
        height_cumulative += s[k]
        plots.append(p)    
    
    plt.ylim((0, 1))  # proportions sum to 1, so the height of the stacked bars is 1
    plt.title('Share of Topics')
    plt.xticks(rotation=90)
    plt.xticks(ind , meetinglist)
    plt.ylabel('Average percentage per topic and policy day')
    
    titles = ['Share of Topic #1', 'Share of Topic #2', 'sadfasdf']
    #for i in range(len(titles)):
    #    titles[i] = titles[i][15:]
    leg = plt.legend(titles, loc=2, fontsize = 'medium')
    for text in leg.get_texts():
        plt.setp(text, weight = 'medium')
    plt.axhline(0.5, color="red", linewidth = 0.5, linestyle = '--')
    
    path ="C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/Latex_MA/Images" #absolute path to save the graphs
    os.chdir(path) 
    plt.savefig("topicmodelling.pdf", bbox_inches='tight')
    
    
    