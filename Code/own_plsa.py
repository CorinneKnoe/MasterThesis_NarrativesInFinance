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

#import functions from other python modules
#-------------------------------------------
path ="C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/Code/" #absolute path to the txt files
os.chdir(path)
from topic_modelling import tokenize, removestopwords, albhabetizer



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
    #'''takes a text and returns a list of words as a document'''
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
    def __init__(self, corpus, number_of_topics, max_iter, model_path):
        self.n_d = len(corpus.documents) #no of documents
        self.n_w = len(corpus.vocabulary) #no of unique words in corpus
        self.n_t = number_of_topics
        self.max_iter = max_iter
        self.model_path = model_path
        self.L = 0.0 # log-likelihood
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

		# P(z|w,d)
        self.p_z_dw = np.zeros([self.n_d, self.n_w, self.n_t], dtype = np.float) #random initialization!
		# P(z|d)
        np.random.seed(123) #seed for random number, so results can be reproduced
        self.p_z_d = np.random.random(size=[self.n_d, self.n_t])
        for di in range(self.n_d):
            normalize(self.p_z_d[di]) #make random numbers for each document over all topics sum to 1
		# P(w|z)
        np.random.seed(456) #seed for random number, so results can be reproduced
        self.p_w_z = np.random.random(size = [self.n_t, self.n_w])
        for zi in range(self.n_t):
            normalize(self.p_w_z[zi])

    def log_likelihood(self): #needs to be optimized with matrix calculation
        L = 0
        for di in range(self.n_d): #no of documents
            for wi in range(self.n_w): #no of unique words
                sum1 = 0
                for zi in range(self.n_t): #no of topics
                    sum1 = sum1 + self.p_z_d[di, zi] * self.p_w_z[zi, wi]
                L = L + self.n_w_d[di, wi] * np.log(sum1)
        return L
    
    def log_likelihood2(self):
        '''likelihood function with array multiplication instead of for loops'''
        multi = np.dot(self.p_z_d, self.p_w_z) #multiplication across all topics
        summulti = self.n_w_d * np.log(multi) #element wise multiplication of two matrices: result from before and document-word count
        L = np.sum(summulti) #add up all elements from resulting matrix
        return L

    def print_p_z_d(self):
        filename = self.model_path + "p_z_d.txt"
        f = open(filename, "w")
        for di in range(self.n_d): #no of documents
            f.write("Doc #" + str(di) +":")
            for zi in range(self.n_t): #no of topic
                f.write(" "+self.p_z_d[di, zi]) #P(z|d), weights on topics per document
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
        filename = self.model_path + "top_words.txt"
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

    def train(self):
        print("Training...")
        for i_iter in range(self.max_iter): #maximal iteration instead of stopping rule? a bit ridiculous?

			# likelihood
            self.L = self.log_likelihood2() #run loglikelihood

            #self.print_top_words(10) #print ten topwords to txt file

            print("Iter " + str(i_iter) + ", L=" + str(self.L)) #print likelihood number of iteration

# =============================================================================
# 			print("E-Step...")
# 			for di in range(self.n_d): #no of documents
# 				for wi in range(self.n_w): #no of words
# 					sum_zk = np.zeros([self.n_t], dtype = float) #range of zeros for k topics
# 					for zi in range(self.n_t): #no of topics, happens accross all topics per document
# 						sum_zk[zi] = self.p_z_d[di, zi] * self.p_w_z[zi, wi] #construct a vector of sums by multiplying row and colum of two matrices
# 					sum1 = np.sum(sum_zk) #sum over all
# 					if sum1 == 0: #when all topics do not show up at all in a document
# 						sum1 = 1
# 					for zi in range(self.n_t):
# 						self.p_z_dw[di, wi, zi] = sum_zk[zi] / sum1 
# 
# 			print("M-Step...")
# 			# update P(z|d) #pi
# 			for di in range(self.n_d):
# 				for zi in range(self.n_t): #no of topics
# 					sum1 = 0.0
# 					sum2 = 0.0
# 					for wi in range(self.n_w): #all words per topic an document
# 						sum1 = sum1 + self.n_w_d[di, wi] * self.p_z_dw[di, wi, zi] #no of words times E-step
# 						sum2 = sum2 + self.n_w_d[di, wi] #total word count per document and topic
# 					if sum2 == 0: #when none of the words show up
# 						sum2 = 1
# 					self.p_z_d[di, zi] = sum1 / sum2
# 
# 			# update P(w|z) #p(w|theta)
# 			for zi in range(self.n_t): #for each topic
# 				sum2 = np.zeros([self.n_w], dtype = np.float) #series of zeroes, enough for each word
# 				for wi in range(self.n_w): #for each word
# 					for di in range(self.n_d): #for each document
# 						sum2[wi] = sum2[wi] + self.n_w_d[di, wi] * self.p_z_dw[di, wi, zi]
# 				sum1 = np.sum(sum2)
# 				if sum1 == 0:
# 					sum1 = 1
# 				for wi in range(self.n_w):
# 					self.p_w_z[zi, wi] = sum2[wi] / sum1  
# =============================================================================

if __name__ == "__main__":
    
    #Read in the text dataframe from a csv
    #-------------------------------------
    path ="C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/FACTIVA_Data/" #absolute path to the txt files
    os.chdir(path)
    os.getcwd()
    #create data frame by reading csv
    textdf = pd.read_csv('FACTIVArticles.csv', sep = ',')
    #turn first colum from string to datetime
    textdf['Date'] = pd.to_datetime(textdf['Date'])
    
    
    #prepare the text data
    #--------------------------------------
    # prepare data and filter out stopwords
    textdf['Article'] = textdf['Article'].apply(tokenize)
    textdf['Article'] = textdf['Article'].apply(removestopwords)
    textdf['Article'] = textdf['Article'].apply(albhabetizer)
    #textdf['Article'] = textdf['Article'].apply(stemmer_porter)
    
    #build the Corpus
    corpus = Corpus()
    for entry in textdf['Article']:
        document = Document(entry)
        corpus.add_document(document)
    corpus.build_vocabulary()
    
    #execute the PLSA
    path ="C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/Code/" #absolute path to where to store the txt files
    number_of_topics = 2 #int(argv[1])
    max_iterations = 1 #int(argv[2])
    
    start_time = time.time()
    plsa = Plsa(corpus, number_of_topics, max_iterations, path)
    print("My program took", time.time() - start_time, "to run")

    start_time = time.time()
    plsa.train()
    print("My program took", time.time() - start_time, "to run")

    start_time = time.time()
    print("My program took", time.time() - start_time, "to run")