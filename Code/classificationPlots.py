# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 11:14:59 2019

@author: corin
"""

'''
new code to put together plots from lists of weights as they
are saved from file own_plsa2_bg.py
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
import matplotlib.patches as mpatches
import seaborn; seaborn.set()
from multiprocessing import Process, Manager
import multiprocessing
import operator
import pickle

if __name__ == "__main__":
    
    
    path ="C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/MA_FinancialData/FED_Data" #absolute path to the txt files
    #path ="C:/Users/11613676/MasterThesis_NarrativesInFinance/MA_FinancialData/FED_Data" #absolute path to the txt files
    os.chdir(path) #setting working directory
    
    #read in the FED data on target rate
    adjustdf = pd.read_csv('adjustments_prep.csv', sep = ',')
    
    #turn date from string into datetime object
    for i in range(len(adjustdf)): #replace date string with date format
        adjustdf.iloc[i,0] = datetime.datetime.strptime(adjustdf.iloc[i,0], '%Y-%m-%d %H:%M:%S')
    
    #list of all meeting dates, to put on x axis, thus as string
    meetinglist = []
    for d in range(len(adjustdf['Date'])):
        meetinglist.append((str(adjustdf.iloc[d, 0])[:10]))
        
    
    lamblist = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    for lamb in lamblist: #for every value of lambda print top ten words of all topics, and create graphs
        name = "Lamb_" + str(lamb).replace('.','_')
    
        with open("C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/Code/Output/weights" + name + ".txt", "rb") as fp:   # Unpickling
            weights = pickle.load(fp)
        with open("C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/Code/Output/docweights" + name + ".txt", "rb") as fp:
            docweights = pickle.load(fp)
            
        #prepare plot - inspiration and code examples from https://de.dariah.eu/tatom/topic_model_visualization.html
        path ="C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/Latex_MA/Images" #absolute path to save the graphs
        #path ="C:/Users/11613676/MasterThesis_NarrativesInFinance/Latex_MA/Images" #absolute path to save the graphs
        os.chdir(path) 
        
        N, K = len(weights), len(weights[0]) #N is numer of meeting dates, K is numer of topics
        ind = np.arange(N)
    
        
        s = []
        for x in range(K):
            t = [] # alist with values for all meetings for one topic
            for entry in weights:
                t.append(entry[x])
            s.append(t)  #a list with K list, each with the weights on the respective topic per meetgin date
              
        seaborn.set(context='paper')    
        fig, ax = plt.subplots(figsize=(14,4))
        ax.plot(s[0])
        ax.fill_between(range(len(meetinglist)), 0, s[0], alpha=1, color=seaborn.color_palette('deep')[0])
        ax.fill_between(range(len(meetinglist)), 1, s[0], alpha=1, color=seaborn.color_palette('deep')[1])
        
        plt.ylim((0, 1))  # proportions sum to 1, so the height of the stacked bars is 1
        plt.title('Share of Topics')
        plt.xticks(rotation=90)
        plt.xticks(ind , meetinglist)
        plt.ylabel('Average percentage per topic and policy day')

        top1 = mpatches.Patch(color=seaborn.color_palette('deep')[0], label='Share of Topic #1')
        top2 = mpatches.Patch(color=seaborn.color_palette('deep')[1], label='Share of Topic #2')
        leg = plt.legend(handles=[top1, top2], loc=2, fontsize = 'medium')
    
        for text in leg.get_texts():
            plt.setp(text, weight = 'medium')
        plt.axhline(0.5, color="red", linewidth = 0.8, linestyle = '--')
        
        for i in range(len(s[0])):
            if s[0][i]>=0.5:
                plt.gca().get_xticklabels()[i].set_color(seaborn.color_palette('deep')[0]) 
            else:
                plt.gca().get_xticklabels()[i].set_color(seaborn.color_palette('deep')[1]) 
       
        #plt.show()
        fig.savefig("plsamodelling_bg"+name+".pdf", bbox_inches='tight')
        
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
        f.savefig("docsplit01_bg"+name+".pdf", bbox_inches='tight')
        
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
        f.savefig("docsplit02_bg"+name+".pdf", bbox_inches='tight')
    
        
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
        fig.savefig("boxplot_"+name+".pdf", bbox_inches='tight')
        


################################## do it all over with a different level for classifying the policy days
                 
        w1sort = sorted(s[0])
        threshhold = w1sort[int(len(w1sort)/2)]               
            
                    
        
        #prepare plot - inspiration and code examples from https://de.dariah.eu/tatom/topic_model_visualization.html
                      
        seaborn.set(context='paper')    
        fig, ax = plt.subplots(figsize=(14,4))
        ax.plot(s[0])
        ax.fill_between(range(len(meetinglist)), 0, s[0], alpha=1, color=seaborn.color_palette('deep')[0])
        ax.fill_between(range(len(meetinglist)), 1, s[0], alpha=1, color=seaborn.color_palette('deep')[1])
        
        plt.ylim((0, 1))  # proportions sum to 1, so the height of the stacked bars is 1
        plt.title('Share of Topics')
        plt.xticks(rotation=90)
        plt.xticks(ind , meetinglist)
        plt.ylabel('Average percentage per topic and policy day')

        top1 = mpatches.Patch(color=seaborn.color_palette('deep')[0], label='Share of Topic #1')
        top2 = mpatches.Patch(color=seaborn.color_palette('deep')[1], label='Share of Topic #2')
        leg = plt.legend(handles=[top1, top2], loc=2, fontsize = 'medium')
    
        for text in leg.get_texts():
            plt.setp(text, weight = 'medium')
        plt.axhline(threshhold, color="red", linewidth = 0.8, linestyle = '--')
        
        for i in range(len(s[0])):
            if s[0][i]>= threshhold:
                plt.gca().get_xticklabels()[i].set_color(seaborn.color_palette('deep')[0]) 
            else:
                plt.gca().get_xticklabels()[i].set_color(seaborn.color_palette('deep')[1]) 
       
        #plt.show()
        fig.savefig("plsamodelling_bg_mod"+name+".pdf", bbox_inches='tight')