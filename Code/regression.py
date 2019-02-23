# -*- coding: utf-8 -*-
"""
CorinneKnoe

created on Feb 19, 2019
This is the script to run the regression with the classified policy days for my 
Master's Thesis. 
"""
import pandas as pd
import os
import numpy as np
import math
import datetime
import matplotlib.pyplot as plt
#%matplotlib inline #for notebooks
import seaborn; seaborn.set()
plt.style.use('seaborn-whitegrid') #whitegrid style for plot
import statsmodels.api as sm
from statsmodels.formula.api import ols


if __name__ == '__main__':
    
    #Reading in data
    #===============
    path ="C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/MA_FinancialData" #absolute path to the txt files
    os.chdir(path) #setting working directory
    
    #read in the FED data on target rate
    financedf = pd.read_csv('financeData.csv', sep = ',')
    
    path ="C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/Code/Output" #absolute path to the txt files
    os.chdir(path) #setting working directory
    
    #read in classification of policy days
    classorigdf = pd.read_csv('AdjustmentsClassifiedPLSAorig.csv', sep = ',')
    
    #read in classification of policy days, lambda = 0.1
    classL0_1 = pd.read_csv('AdjustmentsClassifiedPLSAbgLamb_0_1.csv', sep = ',')
    
    
    #Correct time frame for financial data
    #turn date from string into datetime object
    for i in range(len(financedf)): #replace date string with date format
        financedf.iloc[i,0] = datetime.datetime.strptime(financedf.iloc[i,0], '%Y-%m-%d')
    
    for l in range(len(classorigdf)): #replace date string with date format
        classorigdf.iloc[l,0] = datetime.datetime.strptime(classorigdf.iloc[l,0], '%Y-%m-%d %H:%M:%S')
        classL0_1.iloc[l,0] = datetime.datetime.strptime(classL0_1.iloc[l,0], '%Y-%m-%d %H:%M:%S')
        
    start = datetime.datetime.strptime('01.10.1998', '%d.%m.%Y')
    end = datetime.datetime.strptime('30.09.2018', '%d.%m.%Y')
    financedf = financedf.loc[(financedf.Date >= start) & (financedf.Date < end), :]
    financedf = financedf.reset_index(drop = True)
    
    
    #sort the financial data frame so that newest date first
    financedf = financedf.sort_values(by=['Date'], ascending = False)
    financedf = financedf.reset_index(drop = True)
    
    #build the data frame for regression, calculate the change in interest rate
    regdf = pd.DataFrame()
    regdf['Date'] = financedf['Date']
    regdf.drop(regdf.tail(1).index,inplace=True) # drop last row
    for counter, title in enumerate(['Diff1MT', 'Diff3MT', 'Diff6MT', 'Diff1YT', 
                                     'Diff3YT', 'Diff5YT', 'Diff7YT', 'Diff10YT', 
                                     'Diff20YT', 'Diff30YT'],1):
        regdf[title] = np.array(financedf.iloc[:(len(financedf)-1),counter]) - np.array(financedf.iloc[-(len(financedf)-1):,counter])
    
    #classifications of non policy days
    regdf['NonPolicyDay'] = pd.Series(np.where(financedf.Adjustment.values == 0.0, 1, 0), financedf.index)
    
    #classification of days according to narratives, add to dataframe for regression
    if any(classorigdf['Date'] != classL0_1['Date']):
        raise Exception("The dates fo different classification df do not match!")
    regdf['ClassL=0.0t0'] = ''
    regdf['ClassL=0.0t1'] = ''
    regdf['ClassL=0.1t0'] = ''
    regdf['ClassL=0.1t1'] = ''
    for i in range(len(regdf['Date'])):
        day = regdf.loc[i, 'Date']
        if day in list(classorigdf['Date']):
            rowindex = classorigdf[classorigdf['Date'] == day].index.values.astype(int)[0] #find row of date in classification df
            if classorigdf.loc[rowindex, 'TopicClassification'] == 0:
                regdf.loc[i,'ClassL=0.0t0'] = 1
                regdf.loc[i,'ClassL=0.0t1'] = 0
            if classorigdf.loc[rowindex, 'TopicClassification'] == 1:
                regdf.loc[i,'ClassL=0.0t0'] = 0
                regdf.loc[i,'ClassL=0.0t1'] = 1
            if classL0_1.loc[rowindex, 'ClassificationLamb_0_1'] == 0:
                regdf.loc[i,'ClassL=0.1t0'] = 1
                regdf.loc[i,'ClassL=0.1t1'] = 0
            if classL0_1.loc[rowindex, 'ClassificationLamb_0_1'] == 1:
                regdf.loc[i,'ClassL=0.1t0'] = 0
                regdf.loc[i,'ClassL=0.1t1'] = 1
        else:
            regdf.loc[i, 'ClassL=0.0t0'] = 0
            regdf.loc[i,'ClassL=0.0t1'] = 0
            regdf.loc[i,'ClassL=0.1t0'] = 0
            regdf.loc[i,'ClassL=0.1t1'] = 0
    #check of correctness
    if sum(regdf['NonPolicyDay'] + regdf['ClassL=0.0t0'] + regdf['ClassL=0.0t1']) != len(regdf):
        raise Exception("The data frame for regression is not constructed correctly, check the assignment of topics to policy days (Lambda = 0.0)!")
    if sum(regdf['NonPolicyDay'] + regdf['ClassL=0.1t0'] + regdf['ClassL=0.1t1']) != len(regdf):
        raise Exception("The data frame for regression is not constructed correctly, check the assignment of topics to policy days (Lambda = 0.1)!")
    
    
    #plot the behavior of interest rates on ppolicy days -- original PLSA classification
    # ##################################################################################
    
    plt.style.use('seaborn')
    seaborn.set_context('paper')#, rc={'lines.markeredgewidth': .1})
    fig = plt.figure(figsize=(4,4))
    x = regdf.loc[regdf['NonPolicyDay'] == 0,'Diff3MT']
    y = regdf.loc[regdf['NonPolicyDay'] == 0,'Diff10YT']
    
    plt.scatter(x, y, color=seaborn.color_palette('deep')[0])
    plt.axhline(0, color="black", linewidth = 0.5, linestyle = '-')
    plt.axvline(0, color="black", linewidth = 0.5, linestyle = '-')
    plt.title('Interest rate behavior on policy days')
    plt.xlabel('Change in 3-month rate')
    plt.ylabel('Change in 10-year rate')
    #plt.savefig("C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/Latex_MA/Images/ChangePlot01.pdf", bbox_inches='tight')
    
    #plot the behavior of interest rates on ppolicy days, Narrative One
    plt.style.use('seaborn')
    seaborn.set_context('paper')#, rc={'lines.markeredgewidth': .1})
    fig = plt.figure(figsize=(4,4))
    x = regdf.loc[regdf['ClassL=0.0t0'] == 1,'Diff3MT']
    y = regdf.loc[regdf['ClassL=0.0t0'] == 1,'Diff10YT']
    
    plt.scatter(x, y, color=seaborn.color_palette('deep')[0])
    plt.axhline(0, color="black", linewidth = 0.5, linestyle = '-')
    plt.axvline(0, color="black", linewidth = 0.5, linestyle = '-')
    plt.title('Interest rate behavior on policy days (Narrative one)')
    plt.xlabel('Change in 3-month rate')
    plt.ylabel('Change in 10-year rate')
    #plt.savefig("C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/Latex_MA/Images/ChangePlot02.pdf", bbox_inches='tight')
    
    #plot the behavior of interest rates on ppolicy days, Narrative Two
    plt.style.use('seaborn')
    seaborn.set_context('paper')#, rc={'lines.markeredgewidth': .1})
    fig = plt.figure(figsize=(4,4))
    x = regdf.loc[regdf['ClassL=0.0t1'] == 1,'Diff3MT']
    y = regdf.loc[regdf['ClassL=0.0t1'] == 1,'Diff10YT']
    
    plt.scatter(x, y, color=seaborn.color_palette('deep')[0])
    plt.axhline(0, color="black", linewidth = 0.5, linestyle = '-')
    plt.axvline(0, color="black", linewidth = 0.5, linestyle = '-')
    plt.title('Interest rate behavior on policy days (Narrative two)')
    plt.xlabel('Change in 3-month rate')
    plt.ylabel('Change in 10-year rate')
    #plt.savefig("C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/Latex_MA/Images/ChangePlot03.pdf", bbox_inches='tight')








    #Regression -- classification of PLSA orig
    X = regdf['NonPolicyDay'], regdf['Diff3MT']
    Y = regdf['Diff6MT']
    X = sm.add_constant(X)
    
    model = sm.OLS(Y, X).fit() ## sm.OLS(output, input)
    model.params
    predictions = model.predict(X)
    model.summary()
    
    
    
    NP = regdf.loc[regdf['NonPolicyDay'] == 1,:]
    T0 = regdf.loc[regdf['ClassL=0.0t0'] == 1,:]
    T1 = regdf.loc[regdf['ClassL=0.0t1'] == 1,:]
    len(NP) + len(T0)+ len(T1)
    fit = ols('Diff6MT ~  Diff3MT', data = NP).fit() 
    fit.params
    fit.summary()
    fit = ols('Diff6MT ~  Diff3MT', data = T0).fit() 
    fit.params
    fit.summary()
    fit = ols('Diff6MT ~  Diff3MT', data = T1).fit() 
    fit.params
    fit.summary()    
     
    plt.figure(figsize=(12, 6))
    
    plt.plot(regdf['Diff3MT'], regdf['Diff6MT'], 'o')           # scatter plot showing actual data
    
    plt.plot(regdf['Diff3MT'], predictions, 'r', linewidth=2)   # regression line