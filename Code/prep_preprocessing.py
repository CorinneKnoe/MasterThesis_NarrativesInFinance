# -*- coding: utf-8 -*-
"""
Created on 10.10.2018

@author: coknoe


"""

#install and import packages
#============================
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
import seaborn; seaborn.set()
import re
#import pyprind


#import re
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfTransformer

#import nltk
#nltk.download('stopwords')

#from sklearn.decomposition import LatentDirichletAllocation 
from nltk.corpus import gutenberg
nltk.download('gutenberg')
from wordcloud import WordCloud
from nltk.probability import ConditionalFreqDist
from nltk import FreqDist
from nltk.draw import dispersion_plot
nltk.download('punkt')
from nltk.tokenize import word_tokenize


def preprocessor(text):
    '''apply the nltk tokenizer and remove the stop words, lower case everything'''
    stop = stopwords.words('english')
    text = nltk.word_tokenize(text.strip()) #tokenize the text of the article    
    text = [w.lower() for w in text if w.lower() not in stop]
    text = " ".join(text) #to remove all unnecessary white spaces
    return text.strip()


def tokenizer_porter(text):
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
    #articleno = [] #count the no of articles per adjustment day
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
            #articleno.append((len(articlelist), file)) #stores number of articles appended to df for very date in a list
            for i in range(len(titlelist)): #save content of file as collected in lists in data frame
                articledf = articledf.append([[datelist[i], titlelist[i], articlelist[i], sourcelist[i]]], ignore_index=True)
                articledf.iloc[-50:]
    #name column headers
    articledf.columns = ["Date", "Title", "Article", "Source"]
    
  
    #Save to CSV to use in other projects, etc.
    #==========================================
    #articledf.to_csv("C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/text_Data/FACTIVArticles.csv", index=False, encoding="utf-8") #create a csv to store our data
    

    #Set up data frame with Number of words and articles per day
    #===========================================================
    
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
    stop = stopwords.words('english') #stopwords to be removed 
    wcountdf = pd.DataFrame()
    for i in range(len(adjustdf.loc[:,'Date'])):
        day = adjustdf.loc[i, 'Date']
        dayup = day + datetime.timedelta(days=2) #look for articles day after, needs day = 2 to cover sae and following day, 00:00 is start of day
        daylow = day - datetime.timedelta(days=1) #look for articles day after, and everything between
        slicedf = articledf.loc[(articledf.Date >= daylow) & (articledf.Date <= dayup), :] #filter out articles in correct time frame
        slicedf = slicedf.reset_index(drop=True) #have index of the sliced df start at 0
        noofart = len(slicedf)   #number of articles for a date 
        words = 0
        stoptokens = 0
        for l in range(noofart):
            textprep = nltk.word_tokenize(slicedf.loc[l, 'Article']) #tokenize the text of the article
            words += len(textprep) #adding up the words/tokens in all articles per date    
            stoptokens += len([w for w in textprep if w.lower() not in stop]) #count words without stopwords
        
        wcountdf = wcountdf.append([[day, noofart, words, stoptokens]], ignore_index=True)
        
    #name column headers
    wcountdf.columns = ["Date", "ArticleCount", "TokenCount", "StopTokenCount"]
    sum(wcountdf["TokenCount"])
    
    path ="C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/Latex_MA/Images" #absolute path to save the graphs
    os.chdir(path) 
    
    #Figure of token count per date    
    dateasstring = [str(date)[:10] for date in wcountdf['Date']] #dates not as timestamps so that we have equal spacing
    plt.style.use('seaborn-whitegrid')
    seaborn.set_context('paper')
    fig, ax1 = plt.subplots(figsize=(7,10))
    ax1.barh(dateasstring, list(wcountdf['TokenCount']), 
            label='Number of tokens', color=seaborn.color_palette('deep')[0]) 
    ax1.barh(dateasstring, list(wcountdf['StopTokenCount']), 
            label='Number of tokens w/o stop words', color=seaborn.color_palette('deep')[2])
    
    ax1.set_xlabel('Token Count', color=seaborn.color_palette('deep')[0]) 
    ax1.tick_params('x', labelcolor=seaborn.color_palette('deep')[0])#, labelsize=10)#, labelsize=12)
    #ax1.tick_params('y', labelsize=10)#, labelsize=12)
    #add second axes for line plot
    ax2 = ax1.twiny() #Create a twin Axes sharing the yaxis, here add line plot for article count
    ax2.plot(list(wcountdf['ArticleCount']), dateasstring, linestyle='-', color=seaborn.color_palette('deep')[1], 
             linewidth=2, label='Number of Articles')
    ax2.set_xlabel('Article Count', color=seaborn.color_palette('deep')[1]) 
    ax2.tick_params('x', labelcolor=seaborn.color_palette('deep')[1])#, labelrotation=90)#, labelsize=10)
    #set specifications of plot
    ax1.legend(loc='best', frameon=False) #fontsize=14) #give legend in best corner
    ax2.legend(loc='best', frameon=False)
    ax1.autoscale(tight=True) #removes space above and below bars, tight look
    ax2.autoscale(tight=True) #removes space above and below bars, tight look
    ax1.grid(False) #cancel all grid lines
    ax2.grid(False) #cancel all grid lines
    #ax1.xaxis.grid(color=seaborn.color_palette('deep')[0])#, grid_alpha=0.5, grid_linestyle='.') #add only vertical grid lines
    #ax2.xaxis.grid(color=seaborn.color_palette('deep')[1])#, grid_alpha=1, grid_linestyle='-')
    plt.savefig("tokencount.pdf", bbox_inches='tight')
    
    

    
    #Preprocessing - Cleaning up the text data
    #==========================================
        
    #apply preprocessor to articles in data frame: tokenize and remove stop words
    articledf['Article'] = articledf['Article'].apply(preprocessor)
    articledf['Title'] = articledf['Title'].apply(preprocessor)
    
    #Save to CSV to use in other projects, etc.
    #==========================================
    #articledf.to_csv("C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/text_Data/TokenizedArticles.csv", index=False, encoding="utf-8") #create a csv to store our data
    
    #Counting most common words, explore text data
    #=============================================
    alltext = ''
    for i in range(len(articledf['Article'])):
        alltext += ' ' + articledf.loc[i, 'Article']
        
    alltitle = ''
    for t in range(len(articledf['Title'])):
        alltitle += ' ' + articledf.loc[t, 'Title']
    
    
    #remove all non alphabetic tokens from this text
    alltextlist = [w for w in albhabetizer(alltext).split() if len(w) > 1]
    fdist = FreqDist(alltextlist)
    fdist.most_common(40)
    
    alltitle = [w for w in albhabetizer(alltitle).split() if len(w) > 1]
    fdist = FreqDist(alltitle)
    fdist.most_common(40)
    
    #generate a word cloud
    wcloudalltext = " ".join(alltext).strip() #turn list into string
    wcloud = WordCloud(max_font_size=40, background_color="white").generate(wcloudalltext) #generate a word cloud
    plt.figure(figsize=(40,40))
    plt.imshow(wcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig("C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/Latex_MA/Images/wordcloud.pdf", bbox_inches='tight')
    
    wcloudalltitle = " ".join(alltitle).strip() #turn list into string
    wtitlecloud = WordCloud(max_font_size=40, background_color="white").generate(wcloudalltitle) #generate a word cloud
    plt.figure(figsize=(40,40))
    plt.imshow(wtitlecloud, interpolation="bilinear")
    plt.axis("off")
    
    wcloudtextlist = wcloudalltext.split()
    dispersion_plot(wcloudtextlist, ["interest", "federal reserve", "central bank", "economy"])
    
    words = ['Elinor', 'Marianne', 'Edward', 'Willoughby']
    dispersion_plot(gutenberg.words('austen-sense.txt'), words)
    plt.show()
    
    #Processing documents into tokens (incl. stemming and removing stopwords)
    df['chapter'] = df['chapter'].apply(tokenizer_porter)
    
    #Save to CSV to use in other projects, etc.
    #==========================================
    df.to_csv("got_processed.csv", index=False, encoding="utf-8") #create a csv to store our data
    
    sent = "the the won't, and isn't 99 the dMemoryErrorog dog some other words that we do not care about"
    cfdist = ConditionalFreqDist() #ist ein dictionary
    for word in word_tokenize(sent):
        condition = len(word)
        cfdist[condition][word] += 1
