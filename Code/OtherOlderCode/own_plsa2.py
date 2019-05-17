#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
based on the code of pagelee.sd@gmail.com, zhizhihu.com
Reference:
[1]https://github.com/hitalex/PLSA
----------------------
Adapted and implemented by Corinne Knöpfel
December, 2018
'''
import time
import sys
import os
import glob
import re
import numpy as np
import random
from operator import itemgetter
#import utils.stemmer as stemmer 
import pandas as pd
from nltk.corpus import stopwords
import math
import datetime
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
from multiprocessing import Process, Manager
import multiprocessing
import operator
#import functions from other python modules
#-------------------------------------------
#path ="C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/Code/" #absolute path to the txt files
path ="C:/Users/11613676/MasterThesis_NarrativesInFinance/Code/" #absolute path to the txt files
os.chdir(path)
from prep_preprocessing import tokenize, removestopwords, albhabetizer


#set up functions and PLSA
#-------------------------

def normalize(vec):
	s = sum(vec)
	assert(abs(s) != 0.0) # the sum must not be 0

	for i in range(len(vec)):
		assert(vec[i] >= 0) # element must be >= 0
		vec[i] = vec[i] * 1.0 / s

# di
class Document(object):

    def __init__(self, text):
        '''takes a list of strings and returns the same list of strings, but now 
        the list has a .word function'''
        self.words = []
        wordlist = text 
        for word in wordlist:
            if len(word) > 1:
                self.words.append(word) 

#D
class Corpus(object):
    def __init__(self):
        self.documents = []
        
    def add_document(self, document):
        self.documents.append(document)
        
    def build_vocabulary(self, DisregardedWords):
        discrete_set = set()
        for document in self.documents:
            for word in document.words:
                discrete_set.add(word)
        discrete_set = discrete_set - set(DisregardedWords)
        self.vocabulary = list(discrete_set)
  
def print_p_z_d(model_path, n_d, n_t, p_z_d):
    '''write the topic distribution per document in a word file'''
    filename = model_path + "p_z_d.txt"
    f = open(filename, "w")
    for di in range(n_d): #no of documents
        f.write("Doc #" + str(di) +":")
        for zi in range(n_t): #no of topic
            f.write(" "+ str(p_z_d[di, zi])) #P(z|d), weights on topics per document
        f.write("\n")
    f.close()

def print_p_w_z(model_path, n_t, n_w, p_w_z):
    filename = model_path + "p_w_z.txt"
    f = open(filename, "w")
    for zi in range(n_t): #no of topics
        f.write("Topic #" + str(zi) +":")
        for wi in range(n_w): #no of words
            f.write(" "+p_w_z[zi, wi]) #P(w|z), the probability on a word per topic
        f.write("\n")
    f.close()

def print_top_words(model_path, n_t, corpus, p_w_z, topk):
    filename = model_path + "top_words_plsa.txt"
    f = open(filename, "w")
    for zi in range(n_t): #no of topics
        word_prob = p_w_z[zi,:] #array for each word for one topic
        word_index_prob = []
        for wi in range(len(corpus.vocabulary)):#no of words
            word_index_prob.append([wi, word_prob[wi]]) #append list to list with index number of word and its probability
        word_index_prob = sorted(word_index_prob, key=itemgetter(1), reverse=True) #sort according to second elemetnt of lists of list (probability), largest first
        f.write("-------------\n" + "Topic #" + str(zi) + ":\n")
        for wi in range(topk):
            index = word_index_prob[wi][0] #select index (first element of list of lists)
            prob = word_index_prob[wi][1] #select probability (second element of list of lists)
            f.write(corpus.vocabulary[index] + " " + str(prob) + "\n") #pull word from vocabulary and print with probability
    f.close()
 

def multiprocess_train(randint, corpus, number_of_topics, stoprule, resultdic):
    '''a function that does the training of the EM and the likelihood function 
    but set up so that multiple iterations can run simulatneously'''
    #initialize the variables we need for every run
    #===============================================
    n_d = len(corpus.documents) #no of documents/articles
    n_w = len(corpus.vocabulary) #no of unique words in corpus
    n_t = number_of_topics
    #self.max_iter = max_iter
    stoploop = stoprule
    
    # bag of words
    n_w_d = np.zeros([n_d, n_w], dtype = np.int) #initialze empty matrix/ndarray, documents are lines, words are columns
    vocdic = {k: v for v, k in enumerate(corpus.vocabulary)}
    for di, doc in enumerate(corpus.documents): #give counter and value
        mylist = doc.words
        dic = {k:mylist.count(k) for k in set(mylist) if k in corpus.vocabulary}
        for key in dic.keys():
            index = vocdic[key]
            n_w_d[di, index] = dic[key]  
            
    L = 0.0 # log-likelihood
    diffL = 100000.0 #store difference here, looping value for likelihood function
    # P(z|w,d)
    p_z_dw = np.zeros([n_d, n_w, n_t], dtype = np.float) #empty initialization!
	# P(z|d)
    np.random.seed(randint) #seed for random number, so results can be reproduced
    p_z_d = np.random.random(size=[n_d, n_t])
    for di in range(n_d):
        normalize(p_z_d[di]) #make random numbers for each document over all topics sum to 1
    
    # P(w|z)
    p_w_z = np.random.random(size = [n_t, n_w])
    for zi in range(n_t):
        normalize(p_w_z[zi])
    
    #start run by calculating likelihood
    #===============================================
    print("Training...")
    #for i_iter in range(self.max_iter): #maximal iteration instead of stopping rule? a bit ridiculous?
    i_iter = 0
    while diffL > stoploop: #do until stopping point is reached
        
        i_iter += 1 #just to count the iterations
        
        multi = np.dot(p_z_d, p_w_z) #multiplication across all topics
        if 0 in multi:
            multi[multi == 0] = 1 #canot take log(0), but log(1) = 0
        summulti = n_w_d * np.log(multi) #element wise multiplication of two matrices: result from before and document-word count
        placeholderL = np.sum(summulti) #add up all elements from resulting matrix, #produce Likelihood and hold temporarily so difference to last round can be assessed
        
        diffL = abs(L - placeholderL) #calculate increase in L from last round
        L = placeholderL #store this round's L result
    
                    
       # print("E-Step...")
        p_z_dw = np.einsum('ij,jk->ijk', p_z_d, p_w_z)
        divider = np.einsum('ij,jk->ik', p_z_d, p_w_z)
        divider[divider == 0] = 1 #replace all zero values with 1, cannot divide by zero
        p_z_dw = p_z_dw / divider[:,None]
        p_z_dw = np.transpose(p_z_dw, (0,2,1)) #transpose so depth is document, heigt is words, breadth is topic (switching word and topic to fit original set up)
       
       # print("M-Step...")
        p_z_d = np.einsum('ij,ijk->ik', n_w_d, p_z_dw) #p_z_d
        divider = np.einsum('ij -> i', p_z_d)
        divider[divider == 0] = 1 #replace all zero values with 1, cannot divide by zero
        p_z_d = p_z_d / divider[:,None]
        #check normalization
        if round(sum(p_z_d[0,:]),2) != 1.0:
            raise Exception("Normalization for topic coverage (M-Step) is wrong")
               
        # update P(w|z) #p(w|theta)
        p_w_z = np.einsum('ij,ijk->kj', n_w_d, p_z_dw) #p_z_d
        divider = np.einsum('ij->i', p_w_z)
        divider[divider == 0] = 1 #replace all zero values with 1, cannot divide by zero
        p_w_z = p_w_z / divider[:,None]
        #check normalization
        if round(sum(p_w_z[0,:]),2) != 1.0:
            raise Exception("Normalization for word distributions (M-Step) is wrong")
    
    resultdic[L] = (p_z_d, p_w_z, i_iter)
        

if __name__ == "__main__":
        
    #Read in the text dataframe from a csv
    #-------------------------------------
    #path ="C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/FACTIVA_Data/" #absolute path to the txt files
    path ="C:/Users/11613676/MasterThesis_NarrativesInFinance/FACTIVA_Data/" #absolute path to the txt files
    os.chdir(path)

    #create data frame by reading csv
    textdf = pd.read_csv('TokenizedArticles.csv', sep = ',')
    
    #turn first colum from string to datetime
    #textdf['Date'] = pd.to_datetime(textdf['Date'])
    
    #checking for bigrams and adding them to the list
    #================================================
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
    
    #make some feature selection, discard tokens that appear only 1, 2 or 3 times
    #============================================================================
    #count the words and add up the counts in dictionary
    VocCount = {}
    for Line in range(len(textdf)):
        for Token in textdf.loc[Line , 'Article']:
            if Token not in VocCount.keys():
                VocCount[Token] = 0
            VocCount[Token] += 1
            
    WordsInFrequency = sorted(VocCount.items(), key=operator.itemgetter(1), reverse=False)
    ToBeRemoved = [VocCountItem[0] for VocCountItem in WordsInFrequency if VocCountItem[1] == 1 or VocCountItem[1] == 2 or VocCountItem[1] == 3]
        
    
    #build the Corpus -- 'Article' in textdf is already list of strings
    #--------------------------------------
    corpus = Corpus()
    for entry in textdf['Article']:
        document = Document(entry)
        corpus.add_document(document)
    corpus.build_vocabulary(ToBeRemoved) #here we remove the uncommon words

    #execute the PLSA
    #---------------------------------------
    number_of_topics = 2 #supposed to be 2
    stoprule = 0.01 #supposed to be 0.01
    
# =============================================================================
#     return_dict = {}
#     multiprocess_train(123, corpus, number_of_topics, stoprule, return_dict)
# =============================================================================
      
    
    start_time = time.time()
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    for i in range(3): #how often it is initialized with different values, i.e. how many processes run simultaneously
        p = multiprocessing.Process(target=multiprocess_train, args=(i, corpus, number_of_topics, stoprule, return_dict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
        
    jobs=[]    
    
    for i in range(3,6): #how often it is initialized with different values, i.e. how many processes run simultaneously
        p = multiprocessing.Process(target=multiprocess_train, args=(i, corpus, number_of_topics, stoprule, return_dict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
    print("My parallalized program took", time.time() - start_time, "to run")
    
    keyofbest = max(return_dict.keys()) #find the best rund of all iterations
    
    #print likelihood to txt file
    #filename = "C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/Code/Output/Likelihood_plsa_orig.txt" #absolute path to where to store the txt files
    filename = "C:/Users/11613676/MasterThesis_NarrativesInFinance/Code/Output/Likelihood_plsa_orig.txt" #absolute path to where to store the txt files
    f = open(filename, "w")
    f.write("-------------\n" + "Likelihood of topic model of original PLSA\n")
    f.write("Log Likelihood: " + str(keyofbest) + "\n") 
    f.close() 
    
    #print top ten words of topics to txt file
    path ="C:/Users/11613676/MasterThesis_NarrativesInFinance/Code/Output/" #absolute path to where to store the txt files
    #path ="C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/Code/Output/" #absolute path to where to store the txt files
    print_top_words(path, number_of_topics, corpus, return_dict[keyofbest][1], 50) #printig top 10 words of word distributions
    
    
    # Read in the FEd meetings dates
    #---------------------------------
    p#ath ="C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/MA_FinancialData/FED_Data" #absolute path to the txt files
    path ="C:/Users/11613676/MasterThesis_NarrativesInFinance/MA_FinancialData/FED_Data"
    os.chdir(path) #setting working directory
    
    #read in the FED data on target rate
    adjustdf = pd.read_csv('adjustments_prep.csv', sep = ',')
    
    #turn date from string into datetime object
    for i in range(len(adjustdf)): #replace date string with date format
        adjustdf.iloc[i,0] = datetime.datetime.strptime(adjustdf.iloc[i,0], '%Y-%m-%d %H:%M:%S')
        
    for l in range(len(textdf)): #replace date string with date format
        textdf.iloc[l,0] = datetime.datetime.strptime(textdf.iloc[l,0], '%Y-%m-%d %H:%M:%S')
    

    
    #Returning the topic document distribution
    #Classifying the dates as per topic via average of topci document distribution        
    
    # Stacked Bar plot
    #------------------
    #list of all meeting dates, to put on x axis, thus as string
    meetinglist = []
    for d in range(len(adjustdf['Date'])):
        meetinglist.append((str(adjustdf.iloc[d, 0])[:10]))
        
    
    weights=[]  #take the average of topic porbabilities over all articles per meeting date, this is topic per document distribution
    docweights=[] #weights for every single document, used in lower graph 
    datelist = [] #to check whether right days are being classified, i.e. order here and order in dataframe are same
    for day in list(adjustdf['Date']):
        dayup = day + datetime.timedelta(days=2) #look for articles day after, needs day = 2 to cover sae and following day, 00:00 is start of day
        daylow = day - datetime.timedelta(days=1) #look for articles day after, and everything between
        indexlist = list(textdf.loc[(textdf.Date >= daylow) & (textdf.Date <= dayup), :].index) #all index of df of articles with correct date for that meeting
        line=[]
        for i in indexlist:
            line.append(list(return_dict[keyofbest][0][i])) #here the values are extracted from the matrix p(z|d)
        datelist.append(day)
        weights.append(list(np.mean(line, axis=0))) #only the average over all documents is stored
        docweights.append(line)
        
    #see which topic dominates for a policy day and classify policy day as such  
    classlist = []
    for tup in range(len(weights)):
        dominator = weights[tup].index(max(weights[tup])) #find index of dominating probability, is it topic 0, 1, or ... n
        classlist.append(dominator)
    #add classification to data frame, first  check whether dates are aligned!
    if any(adjustdf["Date"] != datelist):
        raise Exception("Topic classifications are nor allocated correctly to dates in dataframe -- order of days not correct!")
    adjustdf["TopicClassification"] = classlist
    
    #check whether averaging was ok, that is still sums to one across topics per day
    for z in range(len(weights)): #every policy day
        count = 0
        for t in range(len(return_dict[keyofbest][0][0])):  #every topic
            count += weights[z][t]
        if round(count,2) != 1.0:
            raise Exception("The averaged weights for a policy day don't sum to one across topics.")
            
    #save new data frame with classification in csv
    #adjustdf.to_csv("C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/Code/Output/AdjustmentsClassifiedPLSAorig.csv", index=False, encoding="utf-8") #create a csv to store our data
    adjustdf.to_csv("C:/Users/11613676/MasterThesis_NarrativesInFinance/Code/Output/AdjustmentsClassifiedPLSAorig.csv", index=False, encoding="utf-8") #create a csv to store our data
    
    
    #prepare plot - inspiration and code examples from https://de.dariah.eu/tatom/topic_model_visualization.html
    #path ="C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/Latex_MA/Images" #absolute path to save the graphs
    path ="C:/Users/11613676/MasterThesis_NarrativesInFinance/Latex_MA/Images" #absolute path to save the graphs
    os.chdir(path) 
    
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
    
    seaborn.set(context='paper')
    fig = plt.figure(figsize=(14,4))
    height_cumulative = []
    for k in range(K):
        color = seaborn.color_palette('deep')[k]
        if k == 0:
            p = plt.bar(ind, s[k], width, color=color, linewidth=0)
        else:
            p = plt.bar(ind, s[k], width, bottom=height_cumulative, color=color, linewidth=0)
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
    #plt.show()
    fig.savefig("plsamodelling_orig.pdf", bbox_inches='tight')
    
    #here a figure to check whether or not the average is a good indication for classifying a policy day
    #===================================================================================================
    #plot of distribution over the documents per day
    #first 28 days
    seaborn.set(context='paper')
    f, axarr = plt.subplots(7, 4, figsize=(7,10))
        
    for i in range(28):
        if i / 4 < 7:
            c = 6
        if i / 4 < 6:
            c = 5
        if i / 4 < 5:
            c = 4
        if i / 4 < 4:
            c = 3
        if i / 4 < 3:
            c = 2
        if i / 4 < 2:
            c = 1
        if i / 4 < 1:
            c = 0
        
        axarr[c, i%4].bar(list(range(1,len(docweights[i])+1)), sorted([doc[1] for doc in docweights[i]], reverse=True), 
             width = -1, align='edge', linewidth=0, color=seaborn.color_palette('deep')[0]) #xaxis is topic 0, yaxis is topic 1
        axarr[c, i%4].set_title(meetinglist[i])
        axarr[c, i%4].set_ylim([0, 1])
        axarr[c, i%4].set_xlim([1, len(docweights[i])])
        axarr[c, i%4].set_xticks([0, len(docweights[i])//2, len(docweights[i])])
        axarr[c, i%4].set_yticks([0, 0.5, 1])
    plt.tight_layout()
    topic1 = mlines.Line2D([], [], color=seaborn.color_palette('deep')[0], marker='X', linestyle='None',
                          markersize=7, label='Weight of topic 1 in every document of a policy day')
    plt.figlegend(handles=[topic1], loc = 'lower left', ncol=1, prop={'size': 10}, borderaxespad = 0, handletextpad = 0)
    #plt.show()    
    f.savefig("docsplit01_orig.pdf", bbox_inches='tight')
    
    #plot of distribution over the documents per day
    #second 28 days
    seaborn.set(context='paper')
    f, axarr = plt.subplots(7, 4, figsize=(7,10))
    
    for i in range(28, 56):
        if (i-28) / 4 < 7:
            c = 6
        if (i-28) / 4 < 6:
            c = 5
        if (i-28) / 4 < 5:
            c = 4
        if (i-28) / 4 < 4:
            c = 3
        if (i-28) / 4 < 3:
            c = 2
        if (i-28) / 4 < 2:
            c = 1
        if (i-28) / 4 < 1:
            c = 0
            
        axarr[c, i%4].bar(list(range(1,len(docweights[i])+1)), sorted([doc[1] for doc in docweights[i]], reverse=True), 
             width = -1, align='edge', linewidth=0, color=seaborn.color_palette('deep')[0]) #xaxis is topic 0, yaxis is topic 1
        axarr[c, i%4].set_title(meetinglist[i])
        axarr[c, i%4].set_ylim([0, 1])
        axarr[c, i%4].set_xlim([1, len(docweights[i])])
        axarr[c, i%4].set_xticks([0, len(docweights[i])//2, len(docweights[i])])
        axarr[c, i%4].set_yticks([0, 0.5, 1])
    plt.tight_layout()
    topic1 = mlines.Line2D([], [], color=seaborn.color_palette('deep')[0], marker='X', linestyle='None',
                          markersize=7, label='Weight of topic 1 in every document of a policy day')
    plt.figlegend(handles=[topic1], loc = 'lower left', ncol=1, prop={'size': 10}, borderaxespad = 0, handletextpad = 0)
    #plt.show()    
    f.savefig("docsplit02_orig.pdf", bbox_inches='tight')
    
    
    # Box plot - to compare my classification with Ellingsen et al.
    #--------------------------------------------------------------
    #list of all meeting dates that Ellingsen et all classified
    EndoMeetings = ['1998-10-15', '1998-11-17', '1999-06-30', '2001-01-03', '2001-01-31', '2001-03-20', '2001-04-18', '2001-06-27', '2001-08-21', '2001-10-02', '2001-11-06', '2001-12-11']
    ExoMeetings = ['1999-08-24', '1999-11-16', '2000-03-21', '2000-05-16', '2001-05-15', '2001-09-17']
        
    WeightsEndo = []
    for date in EndoMeetings:
        position = meetinglist.index(date)
        WeightsEndo.append(weights[position][0])
    
    WeightsExo = []
    for date in ExoMeetings:
        position = meetinglist.index(date)
        WeightsExo.append(weights[position][0])
    
    #make boxplot of weights of narrative 1
    seaborn.set(context='paper')
    fig = plt.figure(figsize=(7,4))
    plot = plt.boxplot([WeightsEndo, WeightsExo], labels=['endogenous policy days', 'exogenous policy days'], patch_artist=True)
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(plot[element], color=seaborn.color_palette('deep')[0])
    for element in ['medians']:
        plt.setp(plot[element], color=seaborn.color_palette('deep')[1], linewidth=1.3)
    for patch in plot['boxes']:
        patch.set(facecolor=seaborn.color_palette('deep')[0], alpha=1)
    plt.ylabel('Weight of Narrative One')
    fig.savefig("boxplot_orig.pdf", bbox_inches='tight')