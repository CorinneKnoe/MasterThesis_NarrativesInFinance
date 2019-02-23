# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 14:19:19 2019

@author: corin
"""
import nltk
#from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import time
import datetime
import pandas as pd
import os
import math
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
import re  


def dispersion_plot(text, words, xnumbers, xlabels, ignore_case=False, title="Lexical Dispersion Plot"):
    """Generate a lexical dispersion plot.
    :param text: The source text
    :type text: list(str) or enum(str)
    :param words: The target words
    :type words: list of str
    :param ignore_case: flag to set if case should be ignored when searching text
    :type ignore_case: bool
    :xnumbers: a list with numbers where to set the x labels, i.e. at 200'000 words, 400'000 words, etc.
    :xlabels: a list with names/years as to what to call the x labels, 
    done to replace word count with year number
    """

    #try:
    #    from matplotlib import pylab
    #except ImportError:
    #    raise ValueError('The plot function requires matplotlib to be installed.'
           #          'See http://matplotlib.org/')
    text = list(text)
    words.reverse()

    if ignore_case:
        words_to_comp = list(map(str.lower, words))
        text_to_comp = list(map(str.lower, text))
    else:
        words_to_comp = words
        text_to_comp = text

    #points = [(x,y) for x in range(len(text_to_comp)) for y in range(len(words_to_comp)) if text_to_comp[x] == words_to_comp[y]]
    points = [] #finding the points where words are identical, even when words have several word parts
    for x in range(len(text_to_comp)):
        for y in range(len(words_to_comp)):
            wordlength = len(words_to_comp[y].split())
            if " ".join(text_to_comp[x:(x+wordlength)]) == words_to_comp[y]:
                points.append((x,y))
                
    if points:
        x, y = list(zip(*points)) #produces two big tuples
        type(x)
    else:
        x = y = () #empty tuples if there are no matches
    plt.style.use('seaborn')
    seaborn.set_context('paper')#, rc={'lines.markeredgewidth': .1})
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(x, y, '|', color="b", markeredgewidth=0.1, scalex = True) #seaborn sets markeredgewidth to zero, need to specify or nonfilled markers won't show in the plot
    ax.set_xlim(xmin=0, xmax=len(text_to_comp))
    plt.yticks(list(range(len(words))), words, color="b")
    plt.xticks(xnumbers, xlabels)
    plt.ylim(-1, len(words))
    plt.title(title)
    plt.xlabel("Year")
    #plt.savefig("C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/Latex_MA/Images/dispersionplot_PLSAorigTopic.pdf", bbox_inches='tight')




if __name__ == '__main__':
    
    #Read in the text dataframe from a csv
    #-------------------------------------
    path ="C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/FACTIVA_Data/" #absolute path to the txt files
    os.chdir(path)

    #create data frame by reading csv
    tokenizedf = pd.read_csv('TokenizedArticles.csv', sep = ',')
    
     # Read in the FEd meetings dates
    #---------------------------------
    path ="C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/MA_FinancialData/FED_Data" #absolute path to the txt files
    os.chdir(path) #setting working directory
    
    #read in the FED data on target rate
    adjustdf = pd.read_csv('adjustments_prep.csv', sep = ',')
    
    #turn date from string into datetime object
    for i in range(len(adjustdf)): #replace date string with date format
        adjustdf.iloc[i,0] = datetime.datetime.strptime(adjustdf.iloc[i,0], '%Y-%m-%d %H:%M:%S')
        
    for l in range(len(tokenizedf)): #replace date string with date format
        tokenizedf.iloc[l,0] = datetime.datetime.strptime(tokenizedf.iloc[l,0], '%Y-%m-%d %H:%M:%S')
        
      
     # Cut off the dates we don't need from text df, for this we need financial data
    #At the same time, set up data frame with word and article count per date
    ###############################################################################
   
    wcountdf = pd.DataFrame()
    for i in range(len(adjustdf.loc[:,'Date'])):
        day = adjustdf.loc[i, 'Date']
        dayup = day + datetime.timedelta(days=2) #look for articles day after, needs day = 2 to cover sae and following day, 00:00 is start of day
        daylow = day - datetime.timedelta(days=1) #look for articles day after, and everything between
        tokenslicedf = tokenizedf.loc[(tokenizedf.Date >= daylow) & (tokenizedf.Date <= dayup), :] #filter out articles in correct time frame
        noofart = len(tokenslicedf) #number of articles for a date, tokenizedf        
        stoptokens = 0
        tokenslicedf = tokenslicedf.reset_index(drop=True) #have index of the sliced df start at 0
        for l in range(noofart):
            stoptokens += len(tokenslicedf.loc[l, 'Article'].split()) #count toekns of prepared text data
        wcountdf = wcountdf.append([[day, noofart, stoptokens]], ignore_index=True)

    #name column headers
    wcountdf.columns = ["Date", "ArticleCount", "StopTokenCount"]
    sum(wcountdf["StopTokenCount"])
    min(wcountdf["ArticleCount"])
    
    #Counting most common words, explore text data of tokenized text
    #================================================================
    alltext = []  #collect text as counted in wcountdf
    for i in range(len(tokenizedf['Article'])):
        alltext += tokenizedf.loc[i, 'Article'].split()
        
    #dispersion plot of important words
    wordsattick = list(range(100000, 900000, 100000))
    
    [100000, 200000, 400000, 600000, 800000] #where to set xlabels
    
    counter = 0         #get labels for x axis
    index = 0
    labellist = []
    for wc in range(100000, 900000, 100000):
        while counter < wc:
            day = str(wcountdf.iloc[index, 0])[:4]
            counter += wcountdf.iloc[index, 2]
            index += 1
        labellist.append(day)        
    
    dispersion_plot(alltext, ["rate", "fed", "said", "interest", "percent", "year"], wordsattick, labellist)
    dispersion_plot(alltext, ["greenspan", "investors", "officials", "dollar", "trading", "euro"], wordsattick, labellist)
    
    #create a lineplot to assess log likelihood of different PLSA models
    ####################################################################
    plt.style.use('seaborn')
    seaborn.set_context('paper')#, rc={'lines.markeredgewidth': .1})
    fig = plt.figure(figsize=(4,4))
    likelihood = [-6312761.454808442, -6350000.0, -6350000.0, -6350000.0, 
                  -6350000.0, -6350000.0, -6340581.097251067, -6350264.678970301, -6364354.081529713]
    lamb = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8, 0.9]
    
    plt.plot(lamb, likelihood, color=seaborn.color_palette('deep')[0])
    plt.ylim((-6300000, -6375000))  # proportions sum to 1, so the height of the stacked bars is 1
    plt.title('Model Performance of PLSA')
    plt.xlabel('Lambda_B')
    plt.ylabel('Log Likelihood')
    plt.savefig("C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/Latex_MA/Images/ModelLambdaQuality.pdf", bbox_inches='tight')