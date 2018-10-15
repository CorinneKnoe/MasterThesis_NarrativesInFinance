# -*- coding: utf-8 -*-
"""
Created on 10.10.2018

@author: coknoe


"""

#install and import packages
#============================

import pandas as pd
import os
import numpy as np
import math
import datetime





if __name__ == '__main__':
    
    #Reading in data
    #===============
    
    #change path to directory
# =============================================================================
#     #use this when using relative paths  
#     this_path = os.path.abspath(os.path.dirname(__file__)) #take absolute path of file, to construct relative path to data
#     base_path = os.path.dirname(this_path)
#     path = os.path.join(base_path, "MA_FinancialData/FED_Data")
# =============================================================================
    path ="C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/MA_FinancialData/FED_Data" #absolute path to the txt files
    os.chdir(path) #setting working directory
    
    #read in the FED data on target rate
    meetingdf = pd.read_csv('FOMCMeetings.csv', sep = '\t')
    adjustdf = pd.read_csv('adjustments.csv', sep = ',')
    #drop rows if empty
    while math.isnan(adjustdf.iloc[len(adjustdf)-1,1]):
        adjustdf = adjustdf[:-1] #drop last row if it is empty
        
    for i in range(len(meetingdf)): #replace date string with date format
        meetingdf.iloc[i,0] = datetime.datetime.strptime(meetingdf.iloc[i,0], '%d.%m.%Y')
        
    for i in range(len(adjustdf)): #replace date string with date format
        adjustdf.iloc[i,0] = datetime.datetime.strptime(adjustdf.iloc[i,0], '%d/%m/%Y')
    
    #rearrange dataframes to ascending order
    adjustdf.sort_values('Date', inplace = True)
    meetingdf.sort_values('MeetingDate', inplace = True)
    
    #read in the csv files with the treasury yields
    path ="C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/MA_FinancialData/Datastream" #absolute path to the txt files
    os.chdir(path) #setting working directory
    
    #build a dataframe for every file in directory and store in dictionary
    #to every key there is a df with the data for the yield duration in the file
    tydict = {}
    for file in os.listdir(path):
        tydict[file[:-5]] = pd.read_excel(file)#get names of all files in directory
    
    
    #keys are not ordered, here a list with all keys in the order we want them in the final data frame
    yieldorder = ['TRUS1MT', 'TRUS3MT', 'TRUS6MT', 'TRUS1YT', 'TRUS3YT', 'TRUS5YT', 'TRUS7YT', 'TRUS10YT', 'TRUS30YT']   
        
    #create a big dataframe for all yield duration, pull data from dictionary
    treasurydf = pd.DataFrame()  #initiate new data fram
    treasurydf['Date'] = tydict['TRUS1MT'].iloc[:, 0] #we take dates of 1 month rate and put in first column
    for tyield in yieldorder:
        if all(treasurydf['Date'] == tydict[tyield].iloc[:, 0]): #check whether we have identical data range
            treasurydf[tyield] = tydict[tyield].iloc[:,1] #add new column with yield data
        else:
            print("ERROR - Time range does not match") #alert if there is indeed an error in the date range
            break
    
    #add data on target rate to data frame
    treasurydf['TgtLow'] = np.nan
    treasurydf['TgtHigh'] = np.nan
    treasurydf['Adjustment'] = np.nan
    for i in range(len(treasurydf)): #loop through every line of df
        for j in range(len(adjustdf)):
            if treasurydf.iloc[i,0] >= adjustdf.iloc[j,0]: #check whether date is past adjustment date
                treasurydf.loc[treasurydf.index[i], 'TgtLow'] = adjustdf.loc[adjustdf.index[j],'TgtLow'] #if date is past, take values of that adjustment date
                treasurydf.loc[treasurydf.index[i], 'TgtHigh'] = adjustdf.loc[adjustdf.index[j],'TgtHigh']
                if treasurydf.iloc[i,0] == adjustdf.iloc[j,0]: #dummy to indicate whether tgtrate changed that day
                    treasurydf.loc[treasurydf.index[i], 'Adjustment'] = 1
                else:
                    treasurydf.loc[treasurydf.index[i], 'Adjustment'] = 0
    sum(treasurydf['Adjustment'])
    len(adjustdf)
        
        
        

