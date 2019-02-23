#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
based on the code of pagelee.sd@gmail.com, zhizhihu.com
Reference:
[1]https://github.com/hitalex/PLSA
----------------------
Adapted and implemented by Corinne Knöpfel
added a background model to the PLSA estimator
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
#import functions from other python modules
#-------------------------------------------
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
        '''takes a text and returns a list of words as a document, basically a tokenizer'''
        self.words = []
        wordlist = text.split(" ") 
        for word in wordlist:
            if len(word) > 1:
                self.words.append(word) 

#D
class Corpus(object): #gives a list of unique words in all documents
	def __init__(self):
		self.documents = []


	def add_document(self, document): #add an article to list of corpus
		self.documents.append(document)

	def build_vocabulary(self): #create set of unique words in all documents
		discrete_set = set()
		for document in self.documents:
			for word in document.words:
				discrete_set.add(word)
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

def print_top_words(model_path, n_t, corpus, p_w_z, likelihood, label, topk):
    filename = model_path
    f = open(filename, "w")
    f.write("-------------\n" + "Topcis of topic model with background for" + label + "\n")
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
    f.write("-------------\n" + "Likelihood of topic model with background for" + label + "\n")
    f.write("Log Likelihood " + label + ": " + str(likelihood) + "\n") 
    f.close()
 

def multiprocess_train(randint, corpus, backtopic, Lambda, number_of_topics, stoprule, resultdic):
    '''a function that does the training of the EM and the likelihood function 
    but set up so that multiple iterations can run simulatneously, Lambda and backtopic are priors
    with Lambda being a float and backtopic a 1xn_w array'''
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
        dic = {k:mylist.count(k) for k in set(mylist)}
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
    #i_iter = 0
    while diffL > stoploop: #do until stopping point is reached
        
        #i_iter += 1 #just to count the iterations
        
        multi = np.dot(p_z_d, p_w_z) #multiplication across all topics
        multi = Lambda * backtopic + (1-Lambda) * multi #new calculation of likelihood with bg topic
        if 0 in multi:
            multi[multi == 0] = 1 #canot take log(0), but log(1) = 0
        summulti = n_w_d * np.log(multi) #element wise multiplication of two matrices: result from before and document-word count
        placeholderL = np.sum(summulti) #add up all elements from resulting matrix, #produce Likelihood and hold temporarily so difference to last round can be assessed
      
        diffL = abs(L - placeholderL) #calculate increase in L from last round
        L = placeholderL #store this round's L result
    
        #print("Iteration " + str(i_iter) + ", Likelihood: " + str(L) + ", Difference in Likelihood: " + str(diffL))
       
        # print("E-Step...")
       ##topic distribution j=1..k
        p_z_dw = np.einsum('ij,jk->ijk', p_z_d, p_w_z)
        divider = np.einsum('ij,jk->ik', p_z_d, p_w_z)
        divider[divider == 0] = 1 #replace all zero values with 1, cannot divide by zero
        p_z_dw = p_z_dw / divider[:,None]
        p_z_dw = np.transpose(p_z_dw, (0,2,1)) #transpose so depth is document, heigt is words, breadth is topic (switching word and topic to fit original set up)
       
        ##topic distribution background distribution  
        p_z_B = (backtopic * Lambda) / ((backtopic * Lambda) + (1 - Lambda) * divider)
        
       # print("M-Step...")
       #update \pi
        n_w_d_dis = np.einsum('ij, ij -> ij', n_w_d, (1-p_z_B)) #create discounted count 
        p_z_d = np.einsum('ij,ijk->ik', n_w_d_dis, p_z_dw) #p_z_d        
        divider = np.einsum('ij -> i', p_z_d)
        divider[divider == 0] = 1 #replace all zero values with 1, cannot divide by zero
        p_z_d = p_z_d / divider[:,None]
        #check normalization
        if round(sum(p_z_d[0,:]),2) != 1.0:
            raise Exception("Normalization for topic coverage (M-Step) is wrong")
        
        # update P(w|z) #p(w|theta)
        p_w_z = np.einsum('ij,ijk->kj', n_w_d_dis, p_z_dw) #p_z_w
        divider = np.einsum('ij->i', p_w_z)
        divider[divider == 0] = 1 #replace all zero values with 1, cannot divide by zero
        p_w_z = p_w_z / divider[:,None]   
        if round(sum(p_w_z[0,:]),2) != 1.0:
            raise Exception("Normalization for word distributions (M-Step) is wrong")
     
   
    resultdic[L] = p_z_d, p_w_z


if __name__ == "__main__":
        
    #Read in the text dataframe from a csv
    #-------------------------------------
    path ="C:/Users/11613676/MasterThesis_NarrativesInFinance/FACTIVA_Data/" #absolute path to the txt files
    os.chdir(path)

    #create data frame by reading csv
    textdf = pd.read_csv('TokenizedArticles.csv', sep = ',')
    
    #turn first colum from string to datetime
    #textdf['Date'] = pd.to_datetime(textdf['Date'])
    
    #build the Corpus
    #--------------------------------------
    corpus = Corpus()
    for entry in textdf['Article']:
        document = Document(entry)
        corpus.add_document(document)
    corpus.build_vocabulary()
    
    #build the background model (unigram language model) -- prior
    #-------------------------------------------------------------
    n_w = len(corpus.vocabulary) #no of unique words in corpus
    t_backg = np.zeros([n_w], dtype = np.float) #initialize with zeros
    #count the occurence of each word in all documents
    vocdic = {k: v for v, k in enumerate(corpus.vocabulary)} #dictionary for storing index
    countdic = {k: 0 for  k in corpus.vocabulary} #dictionary for storing counts of words
    for document in corpus.documents:
        for word in document.words:
            countdic[word] += 1
    for key in vocdic.keys():
        index = vocdic[key]
        t_backg[index] = countdic[key] 
    normalize(t_backg) #turn it into probability distribution by normalizing it
    
    #print top ten words of background topic
    filename = "C:/Users/11613676/MasterThesis_NarrativesInFinance/Code/Output/top_words_background.txt" #absolute path to where to store the txt files
    f = open(filename, "w")
    word_index_prob = []
    for wi in range(len(corpus.vocabulary)):#no of words
        word_index_prob.append([wi, t_backg[wi]]) #append list to list with index number of word and its probability
    word_index_prob = sorted(word_index_prob, key=itemgetter(1), reverse=True) #sort according to second elemetnt of lists of list (probability), largest first
    f.write("-------------\n" + "Background Topic #:\n")
    for wi in range(10):
        index = word_index_prob[wi][0] #select index (first element of list of lists)
        prob = word_index_prob[wi][1] #select probability (second element of list of lists)
        f.write(corpus.vocabulary[index] + " " + str(prob) + "\n") #pull word from vocabulary and print with probability
    f.close() 
    
    
    #define fraction of documents that-s generated by background topic -- prior
    #---------------------------------------------------------------------------

    #lamblist = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #lamblist = [0.1]
    lamblist = [0.2, 0.3, 0.4, 0.5, 0.6]


        
    #execute the PLSA
    #---------------------------------------
    number_of_topics = 2 #supposed to be 2
    stoprule = 0.1 #supposed to be 0.1
    
# =============================================================================
#     start_time = time.time()
#     return_dict = {}
#     multiprocess_train(123, corpus, t_backg, lamblist[0], number_of_topics, stoprule, return_dict)    
#     print("My normal program took", time.time() - start_time, "to run")
# =============================================================================
    
    
    start_time = time.time()
      
    #here multiprocessing starts
    final_dic = {}
    for lamb in lamblist:
        
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        jobs = []
        
        for i in range(3):#how often it is initialized with different values, i.e. how many processes run simultaneously
            p = multiprocessing.Process(target=multiprocess_train, args=(i, corpus, t_backg, lamb, number_of_topics, stoprule, return_dict))
            jobs.append(p)
            p.start()
        
        for proc in jobs:
            proc.join()
        
        jobs=[]    
        
        for i in range(3,6): #how often it is initialized with different values, i.e. how many processes run simultaneously
            p = multiprocessing.Process(target=multiprocess_train, args=(i, corpus, t_backg, lamb, number_of_topics, stoprule, return_dict))
            jobs.append(p)
            p.start()
    
        for proc in jobs:
            proc.join()
            
        keyofbest = max(return_dict.keys())
        
        final_dic[lamb] = (keyofbest, return_dict[keyofbest]) #append best run for everz lambda
    
    print("My parallalized program took", time.time() - start_time, "to run")
    
    
    
    # Read in the FEd meetings dates
    #---------------------------------
    path ="C:/Users/11613676/MasterThesis_NarrativesInFinance/MA_FinancialData/FED_Data" #absolute path to the txt files
    os.chdir(path) #setting working directory
    
    #read in the FED data on target rate
    adjustdf = pd.read_csv('adjustments_prep.csv', sep = ',')
    
    #turn date from string into datetime object
    for i in range(len(adjustdf)): #replace date string with date format
        adjustdf.iloc[i,0] = datetime.datetime.strptime(adjustdf.iloc[i,0], '%Y-%m-%d %H:%M:%S')
        
    for l in range(len(textdf)): #replace date string with date format
        textdf.iloc[l,0] = datetime.datetime.strptime(textdf.iloc[l,0], '%Y-%m-%d %H:%M:%S')      
    
    # Stacked Bar plot / this remains the same for all lambda
    #------------------
    #list of all meeting dates, to put on x axis, thus as string
    meetinglist = []
    for d in range(len(adjustdf['Date'])):
        meetinglist.append((str(adjustdf.iloc[d, 0])[:10]))
    
    #print likelihood and top words of topics to txt file  
    for lamb in lamblist: #for every value of lambda print top ten words of all topics, and create graphs
        name = "Lamb_" + str(lamb).replace('.','_')
        path ="C:/Users/11613676/MasterThesis_NarrativesInFinance/Code/Output/top_words_plsa_bg" + name + ".txt" #absolute path to where to store the txt files
        print_top_words(path, number_of_topics, corpus, final_dic[lamb][1][1], final_dic[lamb][0], name, 10) #printig top 10 words of word distributions
    
    
        #Returning the topic document distribution
        #Classifying the dates as per topic via average of topic document distribution         
        
        weights=[]  #take the average of topic porbabilities over all articles per meeting date, this is topic per document distribution
        docweights=[] #weights for every single document, used in lower graph 
        datelist = [] #to check whether right days are being classified, i.e. order here and order in dataframe are same
        for day in list(adjustdf['Date']):
            dayup = day + datetime.timedelta(days=2) #look for articles day after, needs day = 2 to cover sae and following day, 00:00 is start of day
            daylow = day - datetime.timedelta(days=1) #look for articles day after, and everything between
            indexlist = list(textdf.loc[(textdf.Date >= daylow) & (textdf.Date <= dayup), :].index) #all index of df of articles with correct date for that meeting
            line=[]
            for i in indexlist:
                line.append(list(final_dic[lamb][1][0][i])) #here the values are extracted from the matrix p(z|d)
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
        adjustdf["Classification" + name] = classlist #add classificaton to data frame
        
        #check whether averaging was ok, that is still sums to one across topics per day
        for z in range(len(weights)): #every policy day
            count = 0
            for t in range(len(final_dic[lamb][1][0][0])):  #every topic
                count += weights[z][t]
            if round(count,2) != 1.0:
                raise Exception("The averaged weights for a policy day don't sum to one across topics.")
                
        
        #prepare plot - inspiration and code examples from https://de.dariah.eu/tatom/topic_model_visualization.html
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
        #plt.show()
        fig.savefig("plsamodelling_bg"+name+".pdf", bbox_inches='tight')
        
        #here a figure to check whether or not the average is a good indication for classifying a policy day
        #===================================================================================================
        #plot of distribution over the documents per day
        #first 28 days
        f, axarr = plt.subplots(7, 4, figsize=(7,10))
        plt.style.use('seaborn-whitegrid')
        #plt.style.use('classic')
        seaborn.set_context('paper')
        
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
                
            classlist = [] #make a list of classifications, to attribute color either to topic 0 being dominant, or topic 2 being dominant
            for w in docweights[i]:
                classlist.append(w.index(max(w)))
                
            axarr[c, i%4].scatter([doc[0] for doc in docweights[i]], [doc[1] for doc in docweights[i]], marker='X', label='Topic 1', 
                 color=[seaborn.color_palette('deep')[c] for c in classlist]) #xaxis is topic 0, yaxis is topic 1
            axarr[c, i%4].set_title(meetinglist[i])
            axarr[c, i%4].set_ylim([0, 1])
            axarr[c, i%4].set_xlim([0, 1])
            axarr[c, i%4].set_xticks([0, 0.5, 1])
            axarr[c, i%4].set_yticks([0, 0.5, 1])
        plt.tight_layout()
        topic1 = mlines.Line2D([], [], color=seaborn.color_palette('deep')[0], marker='X', linestyle='None',
                              markersize=7, label='Document with topic 1 dominant')
        topic2 = mlines.Line2D([], [], color=seaborn.color_palette('deep')[1], marker='X', linestyle='None',
                              markersize=7, label='Document with topic 2 dominant')
        plt.figlegend(handles=[topic1, topic2], loc = 'lower left', ncol=2, prop={'size': 10}, borderaxespad = 0, handletextpad = 0)
        #plt.show()    
        f.savefig("docsplit01_bg"+name+".pdf", bbox_inches='tight')
        
        #plot of distribution over the documents per day
        #second 28 days
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
                
            classlist = [] #make a list of classifications, to attribute color either to topic 0 being dominant, or topic 2 being dominant
            for w in docweights[i]:
                classlist.append(w.index(max(w)))
                
            axarr[c, i%4].scatter([doc[0] for doc in docweights[i]], [doc[1] for doc in docweights[i]], marker='X', label='Topic 1', 
                 color=[seaborn.color_palette('deep')[c] for c in classlist]) #xaxis is topic 0, yaxis is topic 1
            axarr[c, i%4].set_title(meetinglist[i])
            axarr[c, i%4].set_ylim([0, 1])
            axarr[c, i%4].set_xlim([0, 1])
            axarr[c, i%4].set_xticks([0, 0.5, 1])
            axarr[c, i%4].set_yticks([0, 0.5, 1])
        plt.tight_layout()
        topic1 = mlines.Line2D([], [], color=seaborn.color_palette('deep')[0], marker='X', linestyle='None',
                              markersize=7, label='Document with topic 1 dominant')
        topic2 = mlines.Line2D([], [], color=seaborn.color_palette('deep')[1], marker='X', linestyle='None',
                              markersize=7, label='Document with topic 2 dominant')
        plt.figlegend(handles=[topic1, topic2], loc = 'lower left', ncol=2, prop={'size': 10}, borderaxespad = 0, handletextpad = 0)
        #plt.show()  
        f.savefig("docsplit02_bg"+name+".pdf", bbox_inches='tight')
    
    #save new data frame with classification in csv
    adjustdf.to_csv("C:/Users/11613676/MasterThesis_NarrativesInFinance/Code/Output/AdjustmentsClassifiedPLSAbg"+name+".csv", index=False, encoding="utf-8") #create a csv to store our data
        

