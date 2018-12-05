# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 18:48:42 2018

@author: corin
"""
import numpy as np
import time
np.random.seed(42)
random.random

import random
np.random.seed(123)
np.random.random(size=[3, 5])
random.seed(123)
np.random.random(size = [3, 5])

random.seed(123)
np.random.random(size=[3, 5])
random.seed(123)
np.random.random(size = [3, 5])

pzd = np.array([[1, 2, 3], [11, 12, 13], [21, 22, 23]])
pwz = np.array([[1,1,1,1], [10,10,10,10], [100,100,100,100]])
dwc = np.array([[11, 12, 13, 14], [111, 112, 113, 114], [221, 222, 223, 224]])
    
def log_likelihood(pzd, pwz, dwc): #needs to be optimized with matrix calculation
		L = 0
		for di in range(len(pzd)): #no of documents
			for wi in range(len(pwz[0])): #no of unique words
				sum1 = 0
				for zi in range(len(pwz)): #no of topics
					sum1 = sum1 + pzd[di, zi] * pwz[zi, wi]
				L = L + dwc[di, wi] * np.log(sum1)
		return L
    
def log_likelihood2(pzd, pwz, dwc):
    '''likelihood function with array multiplication instead of for loops'''
    multi = np.dot(pzd, pwz)
    summulti = dwc * np.log(multi)
    L = np.sum(summulti)
    return L
    
    
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
        multi = np.dot(self.p_z_d, self.p_w_z)
        summulti = self.n_w_d * multi
        L = np.sum(summulti)
        return L
        
    
    
    log_likelihood(pzd, pwz, dwc)
    log_likelihood2(pzd, pwz, dwc)
    
    
mylist = ["the", "after", "and", "the", "after", "after", "end", "the"]
dic = {k:mylist.count(k) for k in set(mylist)}
for key in dic.keys():
for index in len(corpus.vocabulary):
    word = corpus.vocabulary[index]
    if word in dic.keys():
        n_w_d[di, index] = dic[word]
    
    np.zeros([2500, 23000], dtype = np.int)
    
    
    pzd = np.array([[1, 2, 3], [11, 12, 13], [21, 22, 23]])
    pwz = np.array([[1,1,1,1], [10,10,10,10], [100,100,100,100]])
    np.dot(pzd, pwz)
    pzd * 10
    sum(sum(pzd))
    muster = np.array([[1, 2, 3], [11, 12, 13], [21, 22, 23]])
    muster2 = np.array([[100,200,300], [1000,2000,3000], [5,50,500]])
    sum(sum(muster * muster2))
    np.sum(muster * muster2)
    
    start_time = time.time()
    n_d = len(corpus.documents) #no of documents
    n_w = len(corpus.vocabulary) 
    n_w_d = np.zeros([n_d, n_w], dtype = np.int) 
    vocdic = {k: v for v, k in enumerate(corpus.vocabulary)}
    vocdic["smile"]
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
                for index in range(n_w):
                    word = corpus.vocabulary[index]
                    if word in dic.keys():
                        n_w_d[di, index] = dic[word]
    print("My program took", time.time() - start_time, "to run")
             

           
    start_time = time.time()
    n_d = len(corpus.documents) #no of documents
    n_w = len(corpus.vocabulary) 
    n_w_d2 = np.zeros([n_d, n_w], dtype = np.int) 
    for di, doc in enumerate(corpus.documents): #give counter and value
            n_w_di = np.zeros([n_w], dtype = np.int) #for every document an array with as many zeros as there are words in the entire document
            for word in doc.words: #call list with clean words per document, can call .words on document object
                if word in corpus.vocabulary:
                    word_index = corpus.vocabulary.index(word)
                    n_w_di[word_index] = n_w_di[word_index] + 1 #add a count for the word at right position
            n_w_d2[di] = n_w_di #put it in matrix above, count of words per document
# =============================================================================
#             mylist = doc.words
#             dic = {k:mylist.count(k) for k in set(mylist)}
#             for key in dic.keys():
#                 for index in range(n_w):
#                     word = corpus.vocabulary[index]
#                     if word in dic.keys():
#                         n_w_d[di, index] = dic[word]
# =============================================================================
    print("My program took", time.time() - start_time, "to run")

    
        start_time = time.time()
        
        n_d = len(corpus.documents) #no of documents
        n_w = len(corpus.vocabulary) #no of unique words in corpus
        n_w_d = np.zeros([n_d, n_w], dtype = np.int) #initialze empty matrix/ndarray, documents are lines, words are columns
        vocdic = {k: v for v, k in enumerate(corpus.vocabulary)}
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
                index = vocdic[key]
                n_w_d[di, index] = dic[key]
                
        print("My program took", time.time() - start_time, "to run")

    
