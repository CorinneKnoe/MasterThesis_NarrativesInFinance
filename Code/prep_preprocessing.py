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
#import pyprind
import pandas as pd
import os
#import numpy as np
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
    
    #change basepath to directory of txt data
    this_path = os.path.abspath(os.path.dirname(__file__)) #take absolute path of file, to construct relative path to data
    path = os.path.join(this_path, "../FACTIVA_Data")
    #path ="C:\\Users\\corin\\Documents\\Uni\\M.A.HSG\\MA_Arbeit\\MasterThesis_NarrativesInFinance\\FACTIVA_Data"
    os.chdir(path)
    
    
    timetest = "16 June 2018 20:49:00"
    date_time_obj = datetime.datetime.strptime(timetest, '%d %B %Y %X')
    print(date_time_obj)
    
    #pull content from txt files and save in lists
    title = False
    source = False
    day = False
    time = False
    article = False
    timecheck = False
    titlelist = []
    datelist = []
    sourcelist = []
    articlelist =[]
    articletext = ""
    f = open('18Jun_00.txt') 
    for line in f:
        line = line.strip()
        if title:
            titlelist.append(line)
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
            articletext = " ". join([articletext, line]) #collect all lines of the article in one string
            if line == "CO":
                articlelist.append(articletext[-120:])
                articletext = "" #empty variable for next use
                article = False
            if line == "RF":
                articlelist.append(articletext[-120:])
                articletext = "" #empty variable for next use
                article = False
        if line == "HD":
            title = True
        if line == "PD":
            day = True
        if line == "SN":
            source = True
        if line == "LP" or "TD":
            article = True
    f.close()
    print(set(sourcelist))
    #create a data frame
    df = pd.DataFrame()                 #create a dataframe with columns for POV and text
    print(len(articlelist))
    print(len(titlelist))
    for i in range(len(titlelist)):
        df = df.append([[datelist[i], titlelist[i], articlelist[i], sourcelist[i]]], ignore_index=True)
    df.columns = ["Date", "Title", "Article", "Source"] #"article"]
    df.head(10)


    
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
