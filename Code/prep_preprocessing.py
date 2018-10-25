# -*- coding: utf-8 -*-
"""
Created on 10.10.2018

@author: coknoe


"""

#install and import packages
#============================
import nltk
import time
import datetime
#nltk.download()
import pandas as pd
import os, sys
import numpy as np

#import pyprind


import re
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfTransformer
from nltk.stem.porter import PorterStemmer
#import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
#from sklearn.decomposition import LatentDirichletAllocation 

from nltk.probability import ConditionalFreqDist
nltk.download('punkt')
from nltk.tokenize import word_tokenize


def preprocessor(text):
    '''removing all punctuation, non-letter characters and white spaces'''
    text.strip()
    text = (re.sub('[\W]+', ' ', text.lower()))      #remove non-word characters and make text lowercase
    text = (re.sub('[\d]+', '', text)) #to remove numbers [0-9]
    text = (re.sub('\n', ' ', text))
    " ".join(text.split()) #to remove all unnecessary white spaces
    return text.strip()


def tokenizer_porter(text):
    '''Porter stemmer - split text and convert all words back to 
    their stem, e.g. running -> run, return a list of the stemmed words'''
    porter = PorterStemmer()
    stop = stopwords.words('english')
    stem = [porter.stem(word) for word in text.split()]
    stopremoved = [w for w in stem if w not in stop] #removing common stop words as downloaded from nltk package
    return " ".join(stopremoved)



if __name__ == '__main__':
    
    #Reading in data and preparing for preprocessing
    #===============================================
    
    #change path to directory of txt data
# =============================================================================
#     #use this when using relative paths  
#     this_path = os.path.abspath(os.path.dirname(__file__)) #take absolute path of file, to construct relative path to data
#     base_path = os.path.dirname(this_path)
#     path = os.path.join(base_path, "text_Data")
# =============================================================================
    path ="C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/text_Data/" #absolute path to the txt files
    os.chdir(path) #setting working directory
    
    #Reading in text data between Oct 1, 98 and Sep. 30,2018
    #===============================================
    
    #initialiye indicator variables
    title = False
    source = False
    day = False
    time = False
    article = False
    timecheck = False
    double = False
    articleno = [] #count the no of articles per adjustment day
    #create a data frame
    articledf = pd.DataFrame()                 #create a dataframe with columns for date, source, title and article text
    
    #open every file in directory and save content into data frame
    for file in os.listdir(path):  #get names of all files in directory
        titlelist = [] #initialize empty storage space for every txt file
        datelist = []
        sourcelist = []
        articlelist =[]
        articletext = ""
        #print(os.path.join(path,file))
        #with open(os.path.join(path,file), 'r') as infile:
         #   f = infile.read()
        if  datetime.datetime.strptime(file[:-4], '%Y-%m-%d') > datetime.datetime.strptime('1998-10-01', '%Y-%m-%d'): #only read in data if after Oct 1, 1998
            f = open(os.path.join(path,file)) #open file of this iteration of loop
            for line in f: #read through every line of file that was just opened
                line = line.strip()
                if title:
                    titlelist.append(line)
                    if titlelist[-1] in titlelist[:-1]: #check whether exact same title has already been stored, indicates double
                        double = True
                    title = False
                if time: #via timecheck alerted that this line will contain a time, set before timecheck
                    date = " ".join([daycontent,  line]) + ":00" #add day and time string as well as 00 for format
                    datelist.append(datetime.datetime.strptime(date, '%d %B %Y %X')) #transform string into timestamp
                    time = False
                if timecheck: #need to check whether the line is ET, or no ET data is avaialable, set before day!
                    if line == "ET":
                        time = True
                        timecheck = False
                    else:
                        date = " ".join([daycontent,  "00:00:00"]) #no time, just take string for time, add 00:00:00 for time
                        datelist.append(datetime.datetime.strptime(date, '%d %B %Y %X')) #transform string into timestamp
                        timecheck = False
                if day:
                    daycontent = line
                    day = False
                    timecheck = True  #set time to true, to check whether next line is ET or not (not always the case)
                if source:
                    sourcelist.append(line)
                    source = False
                if article:
                    if line != 'TD': #in case article started with marker LP, we don't want to include TD in text
                        articletext = " ". join([articletext, line]).strip() #collect all lines of the article in one string
                        if line == "CO" or line == 'RF' or line == 'IN' or line == 'NS': #possible markers for end of article
                            articlelist.append(articletext[:-2].strip())
                            articletext = "" #empty variable for next use
                            article = False
                            if double: # drop entry if it is double
                                titlelist = titlelist[:-1]
                                datelist = datelist[:-1]
                                sourcelist = sourcelist[:-1]
                                articlelist = articlelist[:-1]
                                double = False
                        if line == 'Related': #end article before related articles are mentioned
                            articlelist.append(articletext[:-7].strip())
                            articletext = "" #empty variable for next use
                            article = False
                            if double: # drop entry if it is double
                                titlelist = titlelist[:-1]
                                datelist = datelist[:-1]
                                sourcelist = sourcelist[:-1]
                                articlelist = articlelist[:-1]
                                double = False
                if line == "HD":
                    title = True
                if line == "PD":
                    day = True
                if line == "SN":
                    source = True
                if line == "LP" or line == "TD":
                    article = True
            f.close()
            articleno.append((len(articlelist), file)) #stores number of articles appended to df for very date in a list
            for i in range(len(titlelist)): #save content of file as collected in lists in data frame
                articledf = articledf.append([[datelist[i], titlelist[i], articlelist[i], sourcelist[i]]], ignore_index=True)
        
    #name column headers
    articledf.columns = ["Date", "Title", "Article", "Source"]
    
#    len(articleno) #to see how many articles there are per file
#    max(articleno)
#    min(articleno)
    
    #Save to CSV to use in other projects, etc.
    #==========================================
    articledf.to_csv("C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/FACTIVA_Data/FACTIVArticles.csv", index=False, encoding="utf-8") #create a csv to store our data
    
    
# =============================================================================
#     articledf.shape
#     articledf.head(20)
#     articledf['Title']
#     articledf.iloc[3,2]
#     articledf.iloc[0,:]
# =============================================================================
    articleno[0][0]
   articledf.iloc[0,1]
   word_tokenize(articledf.iloc[0,1])
   articledf.iloc[0,2]
   word_tokenize(articledf.iloc[0,2])

    #Set up data frame with No of words and articles per day
    #=======================================================
    
    #call FED data
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
    adjustdf.sort_values('Date', inplace = True)
    
    #set start and end dates, cut df to fit windows we want
    start = datetime.datetime.strptime('01.10.1998', '%d.%m.%Y')
    end = datetime.datetime.strptime('01.10.2018', '%d.%m.%Y')
    adjustdf = adjustdf.loc[(adjustdf.Date >= start) & (adjustdf.Date < end), :]
    
    #Set up data frame with word and article count per date
    wcount = pd.DataFrame()
    for i in range(len(adjustdf.loc[:,'Date'])):
        day = adjustdf.loc[0, 'Date']
        dayup = day + datetime.timedelta(days=1) #look for articles day prior
        daylow = day - datetime.timedelta(days=1) #look for articles day after, and everything between
    
    wcount.append([[datelist[i], titlelist[i], articlelist[i], sourcelist[i]]], ignore_index=True)
        
    #name column headers
    wcount.columns = ["Date", "ArticleCount", "WordCount"]
    
   
    
    #Preprocessing - Cleaning up the text data
    #==========================================
    
    #apply preprocessor to speeches in data frame
    df['chapter'] = df['chapter'].apply(preprocessor)
    
    #Processing documents into tokens (incl. stemming and removing stopwords)
    df['chapter'] = df['chapter'].apply(tokenizer_porter)
    
    #Save to CSV to use in other projects, etc.
    #==========================================
    df.to_csv("got_processed.csv", index=False, encoding="utf-8") #create a csv to store our data
    
    sent = "the the won't, and isn't 99 the dog dog some other words that we do not care about"
    cfdist = ConditionalFreqDist() #ist ein dictionary
    for word in word_tokenize(sent):
        condition = len(word)
        cfdist[condition][word] += 1
