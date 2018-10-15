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
#     path = os.path.join(base_path, "FACTIVA_Data")
# =============================================================================
    path ="C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/FACTIVA_Data/" #absolute path to the txt files
    os.chdir(path) #setting working directory
    
    
    #initialiye indicator variables
    title = False
    source = False
    day = False
    time = False
    article = False
    timecheck = False
    double = False
    
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
        
        for i in range(len(titlelist)): #save content of file as collected in lists in data frame
            articledf = articledf.append([[datelist[i], titlelist[i], articlelist[i], sourcelist[i]]], ignore_index=True)
    
    #name column headers
    articledf.columns = ["Date", "Title", "Article", "Source"]
    type(articledf.iloc[0,0])
    
    #Save to CSV to use in other projects, etc.
    #==========================================
    articledf.to_csv("FACTIVAarticles.csv", index=False, encoding="utf-8") #create a csv to store our data
    
    
    articledf.shape
    articledf.head(20)
    articledf['Title']
    articledf.iloc[3,2]
    articledf.iloc[0,:]
    articlelist

    testlist = ['cow']
    testlist[-1] in testlist[:-1]
########################################
    df = pd.DataFrame()                 #create a dataframe with columns for date, source, title and article text
    
    #open every file in directory and save content into data frame
    titlelist = [] #initialize empty storage space for every txt file
    datelist = []
    sourcelist = []
    articlelist =[]
    articletext = ""
    #print(os.path.join(path,file))
    #with open(os.path.join(path,file), 'r') as infile:
     #   f = infile.read()
    f = open("C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/FACTIVA_Data/98Sep_00.txt") 
    for line in f:
        line = line.strip()
        if title:
            titlelist.append(line)
            if titlelist[-1] in titlelist[:-1]: #check whether exact same title has already been stored, indicates double
                print('is double', titlelist[-1])
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
    for i in range(len(titlelist)): #save content as collected in lists in data frame
        df = df.append([[datelist[i], titlelist[i], articlelist[i], sourcelist[i]]], ignore_index=True)
        len(datelist)
    #name column headers
    df.columns = ["Date", "Title", "Article", "Source"]
    df.shape
    df.head(20)
    df['Title']
    len(sourcelist)
    f.close()
    
    l = "".join(line.split()) #remove all white spaces to find title
        if l.isalpha() and l.isupper(): #must be a title then
            title.append(line)
            chapter.append('')
            count += 1
        else:
            chapter[count] = chapter[count] + ' ' + line
    
    #put content of DwD in list, according to index, for title and chapter content
    f = open('dwd_prepared.txt')
    for line in f:
        line = line.strip()
        l = "".join(line.split()) #remove all white spaces to find title
        if l.isalpha() and l.isupper(): #must be a title then
            title.append(line)
            chapter.append('')
            count += 1
        else:
            chapter[count] = chapter[count] + ' ' + line
    f.close()
            
# =============================================================================
#     print(len(title)) #checking whether all went well
#     print(len(chapter))
#     print(title[272])  #checking on first chapter of DWD
#     print(chapter[272])
# =============================================================================
    
    #make a second column for POV, to replace descriptive chapter titles with names
    title2 = list()
    descriptivetitles = ['the prophet', 'the captain of guards', 
                         "the kraken's daughter", 'the soiled knight', 
                         'the iron captain', 'the drowned man', 'the queenmaker',
                         'alayne', 'the reaver', 'cat of the canals', 
                         'the princess in the tower', "the merchant's man", 
                         'reek', 'the lost lord', 'the wayward bride', 
                         'the prince of winterfell', 'the watcher', 'the turncloak',
                         "the king's prize", "the blind girl", "a ghost in winterfell",
                         'the queensguard', 'the iron suitor', 'the discarded knight',
                         'the spurned suitor', 'the griffin reborn', 'the sacrifice', 
                         'the ugly little girl', 'the kingbreaker', 'the dragontamer',
                         "the queen's hand", 'the windblown']
    nametitles = ['Aeron Greyjoy', 'Areo Hotah', 'ASHA', 'Arys Oakheart', 
                  'VICTARION', 'Aeron Greyjoy', 'Arianne Martell', 
                  'SANSA', 'VICTARION', 'ARYA', 'Arianne Martell', 
                  'Quentyn Martell', 'THEON', 'Jon Connington', 
                  'ASHA', 'THEON', 'Areo Hotah', 'THEON',
                  'ASHA', 'ARYA', 'THEON', 'Barristan Selmy', 
                  'VICTARION', 'Barristan Selmy', 'Quentyn Martell', 'Jon Connington',
                  'ASHA', 'ARYA', 'Barristan Selmy', 'Quentyn Martell', 
                  'Barristan Selmy', 'Quentyn Martell',]
    #len(nametitles) == len(descriptivetitles) #check
    for entry in title:
        if entry.lower() in descriptivetitles:
            index = descriptivetitles.index(entry.lower())
            entry = nametitles[index]
        title2.append(entry)       
    #print(len(title2)) #check
    #print(len(title)) #check
    
    
    df = pd.DataFrame()                 #create a dataframe with columns for POV and text
    for i in range(len(title)):
        df = df.append([[title[i], title2[i], chapter[i]]], ignore_index=True)
    df.columns = ["origPOV", "namePOV", "chapter"]
    df.head(50)
    #print(set(title2))
   
    #arrange data if needed by alphabetical order of POV
    #=====================================================
    #df = df.sort_values("POV")
    #df = df.reset_index(drop=True)
    
    #Save to CSV to use in other projects, etc.
    #==========================================
    df.to_csv("got.csv", index=False, encoding="utf-8") #create a csv to store our data
    
    #Preprocessing - Cleaning up the text data
    #==========================================
    
    #apply preprocessor to speeches in data frame
    df['chapter'] = df['chapter'].apply(preprocessor)
    
    #Processing documents into tokens (incl. stemming and removing stopwords)
    df['chapter'] = df['chapter'].apply(tokenizer_porter)
    
    #Save to CSV to use in other projects, etc.
    #==========================================
    df.to_csv("got_processed.csv", index=False, encoding="utf-8") #create a csv to store our data
