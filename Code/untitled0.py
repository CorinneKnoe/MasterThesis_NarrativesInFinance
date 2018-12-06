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

a = np.arange(25).reshape(5,5)
b = np.arange(5)
c = np.arange(6).reshape(2,3)
np.einsum('ii->i', a)
np.linalg.multi_dot((p_z_d,p_w_z))


n_d = 3 #no of documents
n_w = 4 #no of unique words in corpus
n_t = 3
p_z_d = np.array([[1,1,1], [2,2,2], [3,3,3]])
p_w_z = np.array([[10,10,10,10], [100,100,100,100], [1000,1000,1000,1000]])
n_w_d = np.array([[1,2,3,4], [5,6,7,8], [1,1,1,1]])

p_z_d = np.array([[1, 13, 7], [2, 4, 8], [12, 3, 3]], dtype = np.float)
p_w_z = np.array([[7,11,5,12], [1,12,2,7], [80,45,36,5]], dtype = np.float)
n_w_d = np.array([[11, 12, 13, 14], [111, 112, 113, 114], [221, 222, 223, 224]])
p_z_dw = np.zeros([n_d, n_w, n_t], dtype = np.float) 

np.random.random(size=[n_d, n_t])
np.einsum('ij,jk->ijk', p_z_d, p_w_z)

np.multiply(p_z_d, p_w_z)
p_z_d * p_w_z


start_time = time.time()
print("E-Step...")
for di in range(n_d): #no of documents
    for wi in range(n_w): #no of words
        sum_zk = np.zeros([n_t], dtype = float) #range of zeros for k topics
        for zi in range(n_t): #no of topics, happens accross all topics per document
            sum_zk[zi] = p_z_d[di, zi] * p_w_z[zi, wi] #construct a vector of sums by multiplying row and colum of two matrices
        sum1 = np.sum(sum_zk) #sum over all
        if sum1 == 0: #when all topics do not show up at all in a document
            sum1 = 1
        for zi in range(n_t):
            p_z_dw[di, wi, zi] = sum_zk[zi] / sum1 
print("My program took", time.time() - start_time, "to run")
                        
                        
start_time = time.time()
print("E-Step...")
p_z_dw2 = np.einsum('ij,jk->ijk', p_z_d, p_w_z)
divider = np.einsum('ij,jk->ik', p_z_d, p_w_z)
p_z_dw2 = p_z_dw2 / divider[:,None]
p_z_dw2 = np.transpose(p_z_dw2, (0,2,1)) #transpose so depth is document, heigt is words, breadth is topic (switching word and topic to fit original set up)
print("My program took", time.time() - start_time, "to run")                    
                    
print("M-Step...")
            # update P(z|d) #pi
for di in range(n_d):
    print('di is ', di)
    for zi in range(n_t): #no of topics
        print('zi is ', zi)
        sum1 = 0.0
        sum2 = 0.0
        for wi in range(n_w): #all words per topic an document
            print('wi is ', wi)
            sum1 = sum1 + n_w_d[di, wi] * p_z_dw[di, wi, zi] #no of words times E-step
            sum2 = sum2 + n_w_d[di, wi] #total word count per document and topic
        if sum2 == 0: #when none of the words show up
            sum2 = 1
        print('sum1 and sum2 are ', sum1, sum2)
        print('dividsion is ', sum1 / sum2)
        p_z_d[di, zi] = sum1 / sum2
        print (p_z_d[di, zi], ' is pzd[ ', di, zi, ' ]' )


print("M-Step...")
p_z_d2 = np.einsum('ij,ijk->ik', n_w_d, p_z_dw) #p_z_d
divider = np.einsum('ij->i', n_w_d)
divider[divider == 0] = 1 #replace all zero values with 1, cannot divide by zero
p_z_d2 = p_z_d2 / divider[:,None]

np.all(p_z_d == p_z_d2)
foo[foo == 0] = m

a = np.array([[0,12,22], [32,32,0], [0,21,22]])
a[a == 0] = 500


# update P(w|z) #p(w|theta)
for zi in range(n_t): #for each topic
    sum2 = np.zeros([n_w], dtype = np.float) #series of zeroes, enough for each word
    for wi in range(n_w): #for each word
        for di in range(n_d): #for each document
            sum2[wi] = sum2[wi] + n_w_d[di, wi] * p_z_dw[di, wi, zi]
    sum1 = np.sum(sum2)
    if sum1 == 0:
        sum1 = 1
    for wi in range(n_w):
        p_w_z[zi, wi] = sum2[wi] / sum1  
        
# update P(w|z) #p(w|theta)
p_w_z2 = np.einsum('ij,ijk->kj', n_w_d, p_z_dw) #p_z_d
divider = np.einsum('ij->i', p_w_z2)
divider[divider == 0] = 1 #replace all zero values with 1, cannot divide by zero
p_w_z2 = p_w_z2 / divider[:,None]                    
                    
                    
p_z_dw2 = np.arange(36).reshape(3,4,3)                    
                    
np.all(p_w_z == p_w_z2)                  
                    













for di in range(n_d): #no of documents
    for wi in range(n_w): #no of words
        sum_zk = np.multiply(p_z_d[di, :], p_w_z[:, wi]) #construct a vector of sums by multiplying row and colum of two matrices
        sum1 = np.sum(sum_zk) #sum over all
        if sum1 == 0: #when all topics do not show up at all in a document
            sum1 = 1
        p_z_dw[di, wi, :] = sum_zk / sum1 



    
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

    np.all(n_w_d == n_w_d2)
