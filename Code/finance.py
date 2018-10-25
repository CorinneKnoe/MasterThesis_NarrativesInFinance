# -*- coding: utf-8 -*-
"""
Created on 10.10.2018

@author: coknoe
produces a csv with all FED and other financial data

"""

#install and import packages
#============================

import pandas as pd
import os
import numpy as np
import math
import datetime
import matplotlib.pyplot as plt
#%matplotlib inline #for notebooks
import seaborn; seaborn.set()
plt.style.use('seaborn-whitegrid') #whitegrid style for plot





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
    path ="C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/MA_FinancialData/Datastream2" #absolute path to the txt files
    os.chdir(path) #setting working directory
    
    #build a dataframe for every file in directory and store in dictionary
    #to every key there is a df with the data for the yield duration in the file
    tydict = {}
    for file in os.listdir(path):
        tydict[file[:-5]] = pd.read_excel(file)#get names of all files in directory
    
    
    #keys are not ordered, here a list with all keys in the order we want them in the final data frame
    yieldorder = ['TRUS1MT', 'TRUS3MT', 'TRUS6MT', 'TRUS1YT', 'TRUS3YT', 'TRUS5YT', 'TRUS7YT', 'TRUS10YT', 'TRUS20CT', 'TRUS30YT']   
    
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
        #inefficient code, runs a couple of minutes
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
    
    #yields are in %, but target rate still in decimal number, make target in percent as well
    treasurydf['TgtLow'] = treasurydf['TgtLow']*100
    treasurydf['TgtHigh'] = treasurydf['TgtHigh']*100

    #to preven unnecessary decimals
    treasurydf = treasurydf.round(3)
    
    #saving in csv
    treasurydf.to_csv("C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/MA_FinancialData/financeData.csv", index=False, float_format='%.3f') #create a csv to store our data
    
    
    #Producing graphs
    #----------------
    path ="C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/Latex_MA/Images" #absolute path to save the graphs
    os.chdir(path)
    os.getcwd()
    
    #set start and end dates, cut df to fit windows we want
    start = datetime.datetime.strptime('01.10.1998', '%d.%m.%Y')
    end = datetime.datetime.strptime('01.10.2018', '%d.%m.%Y')
    tdf = treasurydf.loc[(treasurydf.Date >= start) & (treasurydf.Date < end), :]
    tdf.loc[5214,'TgtLow']
#    sum(tdf['Adjustment']) #total target rate adjustments in the time period
    
#    tdf.loc[tdf.TRUS1MT.notnull()] #find non nan entries for 1month rate
#    tdf.loc[tdf.TRUS7YT.notnull()] #find non nan entries for 7year rate

#   Graph for 1M treasury    
# =============================================================================
#     #Figure of target rate and 1M treasury rate
#     fig = plt.figure()
#     seaborn.set_context('paper')
#     figM1 = plt.figure(figsize=(7,4))
#     ax = plt.axes()
#     plt.plot(tdf['Date'], tdf['TRUS1MT'], '-', label='1M Treasury Yield', 
#              linewidth=1, color=seaborn.color_palette('deep')[0])
#     plt.plot(tdf['Date'], tdf['TgtLow'], linestyle='-', 
#              linewidth=1, label = 'Target Rate', color=seaborn.color_palette('deep')[2])
#     plt.plot(tdf['Date'], tdf['TgtHigh'], linestyle='-', color=seaborn.color_palette('deep')[2], 
#              linewidth=1, label='_nolegend_')
#     ax.legend(framealpha=1, loc='upper right', frameon=False)
#     plt.ylabel('Interest rate in %')
#     plt.savefig("1Mtreasury.pdf", bbox_inches='tight')
# =============================================================================
    
    
    #Figure of target rate and 3M treasury rate
    fig = plt.figure()
    plt.style.use('seaborn-whitegrid')
    #plt.style.use('classic')
    seaborn.set_context('paper')
    figM1 = plt.figure(figsize=(7,4))
    ax = plt.axes()
    plt.plot(tdf['Date'], tdf['TRUS3MT'], '-', label='3M Treasury Yield', 
             linewidth=1, color=seaborn.color_palette('deep')[0])
    plt.plot(tdf['Date'], tdf['TgtLow'], linestyle='-', 
             linewidth=1, label = 'Target Rate', color=seaborn.color_palette('deep')[2])
    plt.plot(tdf['Date'], tdf['TgtHigh'], linestyle='-', color=seaborn.color_palette('deep')[2], 
             linewidth=1, label='_nolegend_')
    #plt.title('Target Rate and 3M Treasury Yield') #no need for title when labeled in latex anyways
    ax.legend(loc='upper right', frameon=False)
    plt.ylabel('Interest rate in %')
    #plt.savefig("1Mtreasury.png", bbox_inches='tight')
    plt.savefig("3Mtreasury.pdf", bbox_inches='tight')
    
#   For three plots for short, middle and long term rates    
# =============================================================================
#     #Figure of target rate and short treasury rates
#     fig = plt.figure()
#     plt.style.use('seaborn-whitegrid')
#     seaborn.set_context('paper')
#     figM1 = plt.figure(figsize=(7,4))
#     ax = plt.axes()
#     plt.plot(tdf['Date'], tdf['TRUS1MT'], '-', label='1M Treasury Yield', 
#              linewidth=1, color=seaborn.color_palette('deep')[1])
#     plt.plot(tdf['Date'], tdf['TRUS6MT'], '-', label='6M Treasury Yield', 
#              linewidth=1, color=seaborn.color_palette('deep')[3])
#     plt.plot(tdf['Date'], tdf['TRUS1YT'], '-', label='1Y Treasury Yield', 
#              linewidth=1, color=seaborn.color_palette('deep')[4])
#     plt.plot(tdf['Date'], tdf['TgtLow'], linestyle='-', 
#              linewidth=1, label = 'Target Rate', color=seaborn.color_palette('deep')[2])
#     plt.plot(tdf['Date'], tdf['TgtHigh'], linestyle='-', color=seaborn.color_palette('deep')[2], 
#              linewidth=1, label='_nolegend_')
#     ax.legend(loc='upper right', frameon=False)
#     plt.ylabel('Interest rate in %')
#     plt.savefig("shorttreasury.pdf", bbox_inches='tight')
#     
#     #Figure of target rate and middle treasury rates
#     fig = plt.figure()
#     ax = plt.axes()
#     plt.style.use('seaborn-whitegrid')
#     seaborn.set_context('paper')
#     figM1 = plt.figure(figsize=(7,4))
#     plt.plot(tdf['Date'], tdf['TRUS3YT'], '-', label='3Y Treasury Yield', 
#              linewidth=1, color=seaborn.color_palette('deep')[1])
#     plt.plot(tdf['Date'], tdf['TRUS5YT'], '-', label='5Y Treasury Yield', 
#              linewidth=1, color=seaborn.color_palette('deep')[3])
#     plt.plot(tdf['Date'], tdf['TRUS7YT'], '-', label='7Y Treasury Yield', 
#              linewidth=1, color=seaborn.color_palette('deep')[4])
#     plt.plot(tdf['Date'], tdf['TgtLow'], linestyle='-', 
#              linewidth=1, label = 'Target Rate', color=seaborn.color_palette('deep')[2])
#     plt.plot(tdf['Date'], tdf['TgtHigh'], linestyle='-', color=seaborn.color_palette('deep')[2], 
#              linewidth=1, label='_nolegend_')
#     ax.legend(loc='upper right', frameon=False)
#     plt.ylabel('Interest rate in %')
#     plt.savefig("middletreasury.pdf", bbox_inches='tight')
#     
#     #Figure of target rate and long treasury rates
#     fig = plt.figure()
#     plt.style.use('seaborn-whitegrid')
#     seaborn.set_context('paper')
#     figM1 = plt.figure(figsize=(7,4))
#     ax = plt.axes()
#     plt.plot(tdf['Date'], tdf['TRUS10YT'], '-', label='10Y Treasury Yield', 
#              linewidth=1, color=seaborn.color_palette('deep')[1])
#     plt.plot(tdf['Date'], tdf['TRUS20CT'], '-', label='20Y Treasury Yield', 
#              linewidth=1, color=seaborn.color_palette('deep')[3])
#     plt.plot(tdf['Date'], tdf['TRUS30YT'], '-', label='30Y Treasury Yield', 
#              linewidth=1, color=seaborn.color_palette('deep')[4])
#     plt.plot(tdf['Date'], tdf['TgtLow'], linestyle='-', 
#              linewidth=1, label = 'Target Rate', color=seaborn.color_palette('deep')[2])
#     plt.plot(tdf['Date'], tdf['TgtHigh'], linestyle='-', color=seaborn.color_palette('deep')[2], 
#              linewidth=1, label='_nolegend_')
#     ax.legend(loc='upper right', frameon=False)
#     plt.ylabel('Interest rate in %')
#     plt.savefig("longtreasury.pdf", bbox_inches='tight')
# =============================================================================
    
    
    #A graph of three subplots displayin the yield curve for short, middle and long term treasury yields
    f, axarr = plt.subplots(3, figsize=(7,10))
    plt.style.use('seaborn-whitegrid')
    #plt.style.use('classic')
    seaborn.set_context('paper')
    
    axarr[0].plot(tdf['Date'], tdf['TRUS1MT'], '-', label='1M Treasury Yield', 
             linewidth=1, color=seaborn.color_palette('deep')[1])
    axarr[0].plot(tdf['Date'], tdf['TRUS6MT'], '-', label='6M Treasury Yield', 
             linewidth=1, color=seaborn.color_palette('deep')[3])
    axarr[0].plot(tdf['Date'], tdf['TRUS1YT'], '-', label='1Y Treasury Yield', 
             linewidth=1, color=seaborn.color_palette('deep')[4])
    axarr[0].plot(tdf['Date'], tdf['TgtLow'], linestyle='-', 
             linewidth=1, label = 'Target Rate', color=seaborn.color_palette('deep')[2])
    axarr[0].plot(tdf['Date'], tdf['TgtHigh'], linestyle='-', color=seaborn.color_palette('deep')[2], 
             linewidth=1, label='_nolegend_')
    axarr[0].set_title('Target Rate and Short Term Treasury Yields') #no need for title when labeled in latex anyways
    axarr[0].legend(loc='upper right', frameon=False)
    axarr[0].set_ylabel('Interest rate in %')
    axarr[0].set_ylim([-0.5, 7])
    
    axarr[1].plot(tdf['Date'], tdf['TRUS3YT'], '-', label='3Y Treasury Yield', 
             linewidth=1, color=seaborn.color_palette('deep')[1])
    axarr[1].plot(tdf['Date'], tdf['TRUS5YT'], '-', label='5Y Treasury Yield', 
             linewidth=1, color=seaborn.color_palette('deep')[3])
    axarr[1].plot(tdf['Date'], tdf['TRUS7YT'], '-', label='7Y Treasury Yield', 
             linewidth=1, color=seaborn.color_palette('deep')[4])
    axarr[1].plot(tdf['Date'], tdf['TgtLow'], linestyle='-', 
             linewidth=1, label = 'Target Rate', color=seaborn.color_palette('deep')[2])
    axarr[1].plot(tdf['Date'], tdf['TgtHigh'], linestyle='-', color=seaborn.color_palette('deep')[2], 
             linewidth=1, label='_nolegend_')
    axarr[1].set_title('Target Rate and Middle Term Treasury Yields') #no need for title when labeled in latex anyways
    axarr[1].legend(loc='upper right', frameon=False)
    axarr[1].set_ylabel('Interest rate in %')
    axarr[1].set_ylim([-0.5, 7])
    
    axarr[2].plot(tdf['Date'], tdf['TRUS10YT'], '-', label='10Y Treasury Yield', 
             linewidth=1, color=seaborn.color_palette('deep')[1])
    axarr[2].plot(tdf['Date'], tdf['TRUS20CT'], '-', label='20Y Treasury Yield \n Constant Maturity', 
             linewidth=1, color=seaborn.color_palette('deep')[3])
    axarr[2].plot(tdf['Date'], tdf['TRUS30YT'], '-', label='30Y Treasury Yield', 
             linewidth=1, color=seaborn.color_palette('deep')[4])
    axarr[2].plot(tdf['Date'], tdf['TgtLow'], linestyle='-', 
             linewidth=1, label = 'Target Rate', color=seaborn.color_palette('deep')[2])
    axarr[2].plot(tdf['Date'], tdf['TgtHigh'], linestyle='-', color=seaborn.color_palette('deep')[2], 
             linewidth=1, label='_nolegend_')
    axarr[2].set_title('Target Rate and Long Term Treasury Yields') #no need for title when labeled in latex anyways
    axarr[2].legend(loc='upper right', frameon=False)
    axarr[2].set_ylabel('Interest rate in %')
    axarr[2].set_ylim([-0.5, 7])
    
    plt.savefig("alltreasury.pdf", bbox_inches='tight')

    
    

