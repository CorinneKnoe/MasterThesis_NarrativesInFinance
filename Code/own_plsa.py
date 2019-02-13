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
from multiprocessing import Process, Manager
import multiprocessing
#import functions from other python modules
#-------------------------------------------
path ="C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/Code/" #absolute path to the txt files
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
    #'''takes a text and returns a list of words as a document, basically a tokenizer'''
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


class Plsa(object):
    def __init__(self, corpus, number_of_topics, stoprule, model_path):
        self.n_d = len(corpus.documents) #no of documents/articles
        self.n_w = len(corpus.vocabulary) #no of unique words in corpus
        self.n_t = number_of_topics
        #self.max_iter = max_iter
        self.stoploop = stoprule
        self.model_path = model_path
# =============================================================================
#         self.L = 0.0 # log-likelihood
#         self.diffL = 1000.0 #store difference here, looping value for likelihood function
# =============================================================================
        self.error_L = 0.0001; # error for each iter
        self.corpus = corpus		
        # bag of words
        self.n_w_d = np.zeros([self.n_d, self.n_w], dtype = np.int) #initialze empty matrix/ndarray, documents are lines, words are columns
        self.vocdic = {k: v for v, k in enumerate(corpus.vocabulary)}
        for di, doc in enumerate(corpus.documents): #give counter and value
# =============================================================================
#             n_w_di = np.zeros([self.n_w], dtype = np.int) #for every document an array with as many zeros as there are words in the entire document
#             for word in doc.words: #call list with clean words per document, can call .words on document object
#                 if word in corpus.vocabulary:
#                     word_index = corpus.vocabulary.index(word)
#                     n_w_di[word_index] = n_w_di[word_index] + 1 #add a count for the word at right position
#             self.n_w_d[di] = n_w_di #put it in matrix above, count of words per document
# =============================================================================
            mylist = doc.words
            dic = {k:mylist.count(k) for k in set(mylist)}
            for key in dic.keys():
                index = self.vocdic[key]
                self.n_w_d[di, index] = dic[key]

# =============================================================================
# 		# P(z|w,d)
#         self.p_z_dw = np.zeros([self.n_d, self.n_w, self.n_t], dtype = np.float) #random initialization!
# 		# P(z|d)
#         np.random.seed(123) #seed for random number, so results can be reproduced
#         self.p_z_d = np.random.random(size=[self.n_d, self.n_t])
#         for di in range(self.n_d):
#             normalize(self.p_z_d[di]) #make random numbers for each document over all topics sum to 1
# 		# P(w|z)
#         np.random.seed(456) #seed for random number, so results can be reproduced
#         self.p_w_z = np.random.random(size = [self.n_t, self.n_w])
#         for zi in range(self.n_t):
#             normalize(self.p_w_z[zi])
# =============================================================================

# =============================================================================
#     def log_likelihood(self): #needs to be optimized with matrix calculation, see below
#         L = 0
#         for di in range(self.n_d): #no of documents
#             for wi in range(self.n_w): #no of unique words
#                 sum1 = 0
#                 for zi in range(self.n_t): #no of topics
#                     sum1 = sum1 + self.p_z_d[di, zi] * self.p_w_z[zi, wi]
#                 L = L + self.n_w_d[di, wi] * np.log(sum1)
#         return L
# =============================================================================
   
    def log_likelihood2(self):
        '''likelihood function with array multiplication instead of for loops'''
        multi = np.dot(self.p_z_d, self.p_w_z) #multiplication across all topics
        if 0 in multi:
            multi[multi == 0] = 1 #log of zero is not defined, log(1)=0
        summulti = self.n_w_d * np.log(multi) #element wise multiplication of two matrices: result from before and document-word count
        L = np.sum(summulti) #add up all elements from resulting matrix
        return L

    def print_p_z_d(self):
        '''write the topic distribution per document in a word file'''
        filename = self.model_path + "p_z_d.txt"
        f = open(filename, "w")
        for di in range(self.n_d): #no of documents
            f.write("Doc #" + str(di) +":")
            for zi in range(self.n_t): #no of topic
                f.write(" "+ str(self.p_z_d[di, zi])) #P(z|d), weights on topics per document
            f.write("\n")
        f.close()

    def print_p_w_z(self):
        filename = self.model_path + "p_w_z.txt"
        f = open(filename, "w")
        for zi in range(self.n_t): #no of topics
            f.write("Topic #" + str(zi) +":")
            for wi in range(self.n_w): #no of words
                f.write(" "+self.p_w_z[zi, wi]) #P(w|z), the probability on a word per topic
            f.write("\n")
        f.close()

    def print_top_words(self, topk):
        filename = self.model_path + "top_words_plsa.txt"
        f = open(filename, "w")
        for zi in range(self.n_t): #no of topics
            word_prob = self.p_w_z[zi,:] #array for each word for one topic
            word_index_prob = []
            for wi in range(self.n_w):#no of words
                word_index_prob.append([wi, word_prob[wi]]) #append list to list with index number of word and its probability
            word_index_prob = sorted(word_index_prob, key=itemgetter(1), reverse=True) #sort according to second elemetnt of lists of list (probability), largest first
            f.write("-------------\n" + "Topic #" + str(zi) + ":\n")
            for wi in range(topk):
                index = word_index_prob[wi][0] #select index (first element of list of lists)
                prob = word_index_prob[wi][1] #select probability (second element of list of lists)
                f.write(self.corpus.vocabulary[index] + " " + str(prob) + "\n") #pull word from vocabulary and print with probability
        f.close()
        
        
    def multiprocess_train(self, randinit, resultdic):
        '''a function that does the training of the EM and the likelihood function 
        but set up so that multiple iterations can run simulatneously'''
        #initialize the variables we need for every run
        #===============================================
        L = 0.0 # log-likelihood
        diffL = 1000.0 #store difference here, looping value for likelihood function
        # P(z|w,d)
        p_z_dw = np.zeros([self.n_d, self.n_w, self.n_t], dtype = np.float) #random initialization!
		# P(z|d)
        np.random.seed(randinit) #seed for random number, so results can be reproduced
        p_z_d = np.random.random(size=[self.n_d, self.n_t])
        for di in range(self.n_d):
            normalize(p_z_d[di]) #make random numbers for each document over all topics sum to 1
		# P(w|z)
        np.random.seed(randinit) #seed for random number, so results can be reproduced
        p_w_z = np.random.random(size = [self.n_t, self.n_w])
        for zi in range(self.n_t):
            normalize(p_w_z[zi])
            
        #start run by calculating likelihood
        #===============================================
        print("Training...")
        #for i_iter in range(self.max_iter): #maximal iteration instead of stopping rule? a bit ridiculous?
        i_iter = 0
        while diffL > self.stoploop: #do until stopping point is reached
            
            i_iter += 1 #just to count the iterations
            
            multi = np.dot(p_z_d, p_w_z) #multiplication across all topics
            if 0 in multi:
                multi[multi == 0] = 1 #canot take log(0), but log(1) = 0
            summulti = self.n_w_d * np.log(multi) #element wise multiplication of two matrices: result from before and document-word count
            placeholderL = np.sum(summulti) #add up all elements from resulting matrix, #produce Likelihood and hold temporarily so difference to last round can be assessed
            
            diffL = abs(L - placeholderL) #calculate increase in L from last round
            L = placeholderL #store this round's L result
            
            
            
            #print("Iter " + str(i_iter) + ", L=" + str(L) + ", diff=" + str(diffL)) #print likelihood number of iteration

# =============================================================================
#             print("E-Step...")
#             for di in range(self.n_d): #no of documents
#                 for wi in range(self.n_w): #no of words
#                     sum_zk = np.zeros([self.n_t], dtype = float) #range of zeros for k topics
#                     for zi in range(self.n_t): #no of topics, happens accross all topics per document
#                         sum_zk[zi] = self.p_z_d[di, zi] * self.p_w_z[zi, wi] #construct a vector of sums by multiplying row and colum of two matrices
#                     sum1 = np.sum(sum_zk) #sum over all
#                     if sum1 == 0: #when all topics do not show up at all in a document
#                         sum1 = 1
#                     for zi in range(self.n_t):
#                         self.p_z_dw[di, wi, zi] = sum_zk[zi] / sum1
# =============================================================================
                        
                        
           # print("E-Step...")
            p_z_dw = np.einsum('ij,jk->ijk', p_z_d, p_w_z)
            divider = np.einsum('ij,jk->ik', p_z_d, p_w_z)
            divider[divider == 0] = 1 #replace all zero values with 1, cannot divide by zero
            p_z_dw = p_z_dw / divider[:,None]
            p_z_dw = np.transpose(p_z_dw, (0,2,1)) #transpose so depth is document, heigt is words, breadth is topic (switching word and topic to fit original set up)

# =============================================================================
#             print("M-Step...")
#             # update P(z|d) #pi
#             for di in range(self.n_d):
#                 for zi in range(self.n_t): #no of topics
#                     sum1 = 0.0
#                     sum2 = 0.0
#                     for wi in range(self.n_w): #all words per topic an document
#                         sum1 = sum1 + self.n_w_d[di, wi] * self.p_z_dw[di, wi, zi] #no of words times E-step
#                         sum2 = sum2 + self.n_w_d[di, wi] #total word count per document and topic
#                     if sum2 == 0: #when none of the words show up
#                         sum2 = 1
#                     self.p_z_d[di, zi] = sum1 / sum2
# =============================================================================
            
           # print("M-Step...")
            p_z_d = np.einsum('ij,ijk->ik', self.n_w_d, p_z_dw) #p_z_d
            divider = np.einsum('ij->i', self.n_w_d)
            divider[divider == 0] = 1 #replace all zero values with 1, cannot divide by zero
            p_z_d = p_z_d / divider[:,None]
            
# =============================================================================
#             # update P(w|z) #p(w|theta)
#             for zi in range(self.n_t): #for each topic
#                 sum2 = np.zeros([self.n_w], dtype = np.float) #series of zeroes, enough for each word
#                 for wi in range(self.n_w): #for each word
#                     for di in range(self.n_d): #for each document
#                         sum2[wi] = sum2[wi] + self.n_w_d[di, wi] * self.p_z_dw[di, wi, zi]
#                 sum1 = np.sum(sum2)
#                 if sum1 == 0:
#                     sum1 = 1
#                 for wi in range(self.n_w):
#                     self.p_w_z[zi, wi] = sum2[wi] / sum1 
# =============================================================================
                    
            # update P(w|z) #p(w|theta)
            p_w_z = np.einsum('ij,ijk->kj', self.n_w_d, p_z_dw) #p_z_d
            divider = np.einsum('ij->i', p_w_z)
            divider[divider == 0] = 1 #replace all zero values with 1, cannot divide by zero
            p_w_z = p_w_z / divider[:,None]   
            
        
        #printing topics to file
        #print("printing top words to file...")
        #self.print_top_words(10) #print ten topwords to txt file when process is finished
        
        resultdic[L] = (p_z_d, p_w_z)
        

if __name__ == "__main__":
    
    # Read in the FEd meetings dates
    #---------------------------------
    path ="C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/MA_FinancialData/FED_Data" #absolute path to the txt files
    os.chdir(path) #setting working directory
    
    #read in the FED data on target rate
    adjustdf = pd.read_csv('adjustments_prep.csv', sep = ',')
    
    #Read in the text dataframe from a csv
    #-------------------------------------
    path ="C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/FACTIVA_Data/" #absolute path to the txt files
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
    
    #execute the PLSA
    #---------------------------------------
    path ="C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/Code/" #absolute path to where to store the txt files
    path = ''
    number_of_topics = 2 #int(argv[1])
    stoprule = 0.1 #int(argv[2])
    
    start_time = time.time()
    plsa = Plsa(corpus, number_of_topics, stoprule, path)
    print("My program took", time.time() - start_time, "to run")


    start_time = time.time()
    for k in [123,123]:
        resultdic = {}
        result = plsa.multiprocess_train(k, resultdic)
    print("My program took", time.time() - start_time, "to run")
    #print(resultdic.keys())

    
    start_time = time.time()
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    for i in [123,123]:#range(1,4):
        p = multiprocessing.Process(target=plsa.multiprocess_train, args=(i,return_dict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
    print("My parallalized program took", time.time() - start_time, "to run")
    
    print(return_dict.keys())
    
    
# =============================================================================
# 
#     
#  
#     
#     #Returning the topic document distribution
#     #Classifying the dates as per topic via average of topci document distribution        
#     
#     # Stacked Bar plot
#     #------------------
#     #list of all meeting dates
#     meetinglist = []
#     for d in range(len(feddf['Date'])):
#         meetinglist.append((str(feddf.iloc[d, 0])[:10]))
#         
#     
#     weights=[]  #take the average of topic porbabilities over all articles per meeting date, this is topic per document distribution
#     docweights=[] #weights for every single document, to check whether averaging messes anything up
#     for day in list(feddf['Date']):
#         dayup = day + datetime.timedelta(days=2) #look for articles day after, needs day = 2 to cover sae and following day, 00:00 is start of day
#         daylow = day - datetime.timedelta(days=1) #look for articles day after, and everything between
#         indexlist = list(textdf.loc[(textdf.Date >= daylow) & (textdf.Date <= dayup), :].index) #all index of df of articles with correct date for that meeting
#         line=[]
#         for i in indexlist:
#             line.append(list(plsa.p_z_d[i])) #here the values are extracted from the matrix p(z|d)
#         weights.append(list(np.mean(line, axis=0))) #only the average over all documents is stored
#         docweights.append(line)
#     
#     #prepare plot - inspiration and code examples from https://de.dariah.eu/tatom/topic_model_visualization.html
#     N, K = len(weights), len(weights[0]) #N is numer of meeting dates, K is numer of topics
#     ind = np.arange(N)
#     width = 0.5 
#     plots = []
#     height_cumulative = np.zeros(N)
#     
#     s = []
#     for x in range(K):
#         t = [] # alist with values for all meetings for one topic
#         for entry in weights:
#             t.append(entry[x])
#         s.append(t)  #a list with K list, each with the weights on the respective topic per meetgin date
#         
#     fig = plt.figure(figsize=(14,4))
#     height_cumulative = []
#     for k in range(K):
#         color = seaborn.color_palette('deep')[k]
#         if k == 0:
#             p = plt.bar(ind, s[k], width, color=color)
#         else:
#             p = plt.bar(ind, s[k], width, bottom=height_cumulative, color=color)
#         height_cumulative += s[k]
#         plots.append(p)    
#     
#     plt.ylim((0, 1))  # proportions sum to 1, so the height of the stacked bars is 1
#     plt.title('Share of Topics')
#     plt.xticks(rotation=90)
#     plt.xticks(ind , meetinglist)
#     plt.ylabel('Average percentage per topic and policy day')
#     
#     titles = ['Share of Topic #1', 'Share of Topic #2', 'sadfasdf']
#     #for i in range(len(titles)):
#     #    titles[i] = titles[i][15:]
#     leg = plt.legend(titles, loc=2, fontsize = 'medium')
#     for text in leg.get_texts():
#         plt.setp(text, weight = 'medium')
#     plt.axhline(0.5, color="red", linewidth = 0.5, linestyle = '--')
#     
#     path ="C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/Latex_MA/Images" #absolute path to save the graphs
#     os.chdir(path) 
#     plt.savefig("plsamodelling.pdf", bbox_inches='tight')
#     
#     
#     #plot of distribution over the documents per day
#     #first 28 days
#     f, axarr = plt.subplots(7, 4, figsize=(7,10))
#     plt.style.use('seaborn-whitegrid')
#     #plt.style.use('classic')
#     seaborn.set_context('paper')
#     
#     for i in range(28):
#         if i / 4 < 7:
#             c = 6
#         if i / 4 < 6:
#             c = 5
#         if i / 4 < 5:
#             c = 4
#         if i / 4 < 4:
#             c = 3
#         if i / 4 < 3:
#             c = 2
#         if i / 4 < 2:
#             c = 1
#         if i / 4 < 1:
#             c = 0
#         axarr[c, i%4].plot(range(len(docweights[i])), [doc[0] for doc in docweights[i]], 'X', label='Topic 1', 
#              markeredgewidth=0.5, color=seaborn.color_palette('deep')[0])
#         axarr[c, i%4].plot(range(len(docweights[i]), 2*len(docweights[i])), [doc[1] for doc in docweights[i]], 'X', label='Topic 2', 
#              markeredgewidth=0.5, color=seaborn.color_palette('deep')[1])
#         axarr[c, i%4].set_title(meetinglist[i])
#         axarr[c, i%4].set_ylim([0, 1])
#         axarr[c, i%4].set_xlim([0, 2*len(docweights[i])])
#         axarr[c, i%4].set_xticks([len(docweights[i])])
#         axarr[c, i%4].set_yticks([0, 0.5, 1])
#     plt.tight_layout()
#     topic1 = mlines.Line2D([], [], color=seaborn.color_palette('deep')[0], marker='X', linestyle='None',
#                           markersize=7, markeredgewidth=.5, label='Topic 1')
#     topic2 = mlines.Line2D([], [], color=seaborn.color_palette('deep')[1], marker='X', linestyle='None',
#                           markersize=7, markeredgewidth=.5, label='Topic 2')
#     plt.figlegend(handles=[topic1, topic2], loc = 'lower left', ncol=2, prop={'size': 10}, borderaxespad = 0, handletextpad = 0)
#         
#     #plot of distribution over the documents per day
#     #second 28 days
#     f, axarr = plt.subplots(7, 4, figsize=(7,10))
#     plt.style.use('seaborn-whitegrid')
#     #plt.style.use('classic')
#     seaborn.set_context('paper')
#     
#     for i in range(28, 56):
#         if (i-28) / 4 < 7:
#             c = 6
#         if (i-28) / 4 < 6:
#             c = 5
#         if (i-28) / 4 < 5:
#             c = 4
#         if (i-28) / 4 < 4:
#             c = 3
#         if (i-28) / 4 < 3:
#             c = 2
#         if (i-28) / 4 < 2:
#             c = 1
#         if (i-28) / 4 < 1:
#             c = 0
#         axarr[c, i%4].plot(range(len(docweights[i])), [doc[0] for doc in docweights[i]], 'X', label='Topic 1', 
#              markeredgewidth=0.5, color=seaborn.color_palette('deep')[0])
#         axarr[c, i%4].plot(range(len(docweights[i]), 2*len(docweights[i])), [doc[1] for doc in docweights[i]], 'X', label='Topic 2', 
#              markeredgewidth=0.5, color=seaborn.color_palette('deep')[1])
#         axarr[c, i%4].set_title(meetinglist[i])
#         axarr[c, i%4].set_ylim([0, 1])
#         axarr[c, i%4].set_xlim([0, 2*len(docweights[i])])
#         axarr[c, i%4].set_xticks([len(docweights[i])])
#         axarr[c, i%4].set_xticklabels([])
#         axarr[c, i%4].set_yticks([0, 0.5, 1])
#     plt.tight_layout()
#     topic1 = mlines.Line2D([], [], color=seaborn.color_palette('deep')[0], marker='X', linestyle='None',
#                           markersize=7, markeredgewidth=.5, label='Topic 1')
#     topic2 = mlines.Line2D([], [], color=seaborn.color_palette('deep')[1], marker='X', linestyle='None',
#                           markersize=7, markeredgewidth=.5, label='Topic 2')
#     plt.figlegend(handles=[topic1, topic2], loc = 'lower left', ncol=2, prop={'size': 10}, borderaxespad = 0, handletextpad = 0)
#     
#     
# 
# 
# =============================================================================
