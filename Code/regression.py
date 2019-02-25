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
import statsmodels.formula.api as smf 
import statsmodels.stats.api as sms
from statsmodels.compat import lzip




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
    classL0_6 = pd.read_csv('AdjustmentsClassifiedPLSAbgLamb_0_6.csv', sep = ',')
    classL0_9 = pd.read_csv('AdjustmentsClassifiedPLSAbgLamb_0_9.csv', sep = ',')
    
    
    #Correct time frame for financial data
    #turn date from string into datetime object
    for i in range(len(financedf)): #replace date string with date format
        financedf.iloc[i,0] = datetime.datetime.strptime(financedf.iloc[i,0], '%Y-%m-%d')
    
    for l in range(len(classorigdf)): #replace date string with date format
        classorigdf.iloc[l,0] = datetime.datetime.strptime(classorigdf.iloc[l,0], '%Y-%m-%d %H:%M:%S')
        classL0_1.iloc[l,0] = datetime.datetime.strptime(classL0_1.iloc[l,0], '%Y-%m-%d %H:%M:%S')
        classL0_6.iloc[l,0] = datetime.datetime.strptime(classL0_6.iloc[l,0], '%Y-%m-%d %H:%M:%S')
        classL0_9.iloc[l,0] = datetime.datetime.strptime(classL0_9.iloc[l,0], '%Y-%m-%d %H:%M:%S')
        
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
    
    for lamb in ['L=0.0','L=0.1', 'L=0.2', 'L=0.3', 'L=0.4', 'L=0.5', 'L=0.6', 
                 'L=0.7', 'L=0.8', 'L=0.9']:
        for topic in ['t0', 't1']:
            regdf['Class'+lamb+topic] = 0
    
    for i in range(len(regdf['Date'])):
        day = regdf.loc[i, 'Date']
        if day in list(classorigdf['Date']):
            rowindex = classorigdf[classorigdf['Date'] == day].index.values.astype(int)[0] #find row of date in classification df
            if classorigdf.loc[rowindex, 'TopicClassification'] == 0:
                regdf.loc[i,'ClassL=0.0t1'] = 1
            if classorigdf.loc[rowindex, 'TopicClassification'] == 1:
                regdf.loc[i,'ClassL=0.0t0'] = 1
            if classL0_1.loc[rowindex, 'ClassificationLamb_0_1'] == 0:
                regdf.loc[i,'ClassL=0.1t1'] = 1
            if classL0_1.loc[rowindex, 'ClassificationLamb_0_1'] == 1:
                regdf.loc[i,'ClassL=0.1t0'] = 1
                
            if classL0_6.loc[rowindex, 'ClassificationLamb_0_2'] == 0:
                regdf.loc[i,'ClassL=0.2t0'] = 1
            if classL0_6.loc[rowindex, 'ClassificationLamb_0_2'] == 1:
                regdf.loc[i,'ClassL=0.2t1'] = 1
            if classL0_6.loc[rowindex, 'ClassificationLamb_0_3'] == 0:
                regdf.loc[i,'ClassL=0.3t0'] = 1
            if classL0_6.loc[rowindex, 'ClassificationLamb_0_3'] == 1:
                regdf.loc[i,'ClassL=0.3t1'] = 1
            if classL0_6.loc[rowindex, 'ClassificationLamb_0_4'] == 0:
                regdf.loc[i,'ClassL=0.4t0'] = 1
            if classL0_6.loc[rowindex, 'ClassificationLamb_0_4'] == 1:
                regdf.loc[i,'ClassL=0.4t1'] = 1
            if classL0_6.loc[rowindex, 'ClassificationLamb_0_5'] == 0:
                regdf.loc[i,'ClassL=0.5t0'] = 1
            if classL0_6.loc[rowindex, 'ClassificationLamb_0_5'] == 1:
                regdf.loc[i,'ClassL=0.5t1'] = 1
            if classL0_6.loc[rowindex, 'ClassificationLamb_0_6'] == 0:
                regdf.loc[i,'ClassL=0.6t0'] = 1
            if classL0_6.loc[rowindex, 'ClassificationLamb_0_6'] == 1:
                regdf.loc[i,'ClassL=0.6t1'] = 1
                
            if classL0_9.loc[rowindex, 'ClassificationLamb_0_7'] == 0:
                regdf.loc[i,'ClassL=0.7t0'] = 1
            if classL0_9.loc[rowindex, 'ClassificationLamb_0_7'] == 1:
                regdf.loc[i,'ClassL=0.7t1'] = 1
            if classL0_9.loc[rowindex, 'ClassificationLamb_0_8'] == 0:
                regdf.loc[i,'ClassL=0.8t0'] = 1
            if classL0_9.loc[rowindex, 'ClassificationLamb_0_8'] == 1:
                regdf.loc[i,'ClassL=0.8t1'] = 1
            if classL0_9.loc[rowindex, 'ClassificationLamb_0_9'] == 0:
                regdf.loc[i,'ClassL=0.9t0'] = 1
            if classL0_9.loc[rowindex, 'ClassificationLamb_0_9'] == 1:
                regdf.loc[i,'ClassL=0.9t1'] = 1
            
         
    #check of correctness
    if sum(regdf['ClassL=0.0t0'] + regdf['ClassL=0.0t1']
    + regdf['ClassL=0.1t0'] + regdf['ClassL=0.1t1'] + regdf['ClassL=0.2t0'] + regdf['ClassL=0.2t1']
    + regdf['ClassL=0.3t0'] + regdf['ClassL=0.3t1'] + regdf['ClassL=0.4t0'] + regdf['ClassL=0.4t1'] 
    + regdf['ClassL=0.5t0'] + regdf['ClassL=0.5t1'] + regdf['ClassL=0.6t0'] + regdf['ClassL=0.6t1']
    + regdf['ClassL=0.7t0'] + regdf['ClassL=0.7t1'] + regdf['ClassL=0.8t0'] + regdf['ClassL=0.8t1']
    + regdf['ClassL=0.9t0'] + regdf['ClassL=0.9t1']) != 10*len(classorigdf):
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
    plt.xlim(-0.65, 0.2)
    plt.savefig("C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/Latex_MA/Images/ChangePlot01.pdf", bbox_inches='tight')
    
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
    plt.xlim(-0.65, 0.2)
    plt.savefig("C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/Latex_MA/Images/ChangePlot02.pdf", bbox_inches='tight')
    
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
    plt.xlim(-0.65, 0.2)
    plt.savefig("C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/Latex_MA/Images/ChangePlot03.pdf", bbox_inches='tight')

        #plot the behavior of interest rates on ppolicy days -- lambda = 0.1 PLSA classification
    # ##################################################################################
    
    #plot the behavior of interest rates on ppolicy days, Narrative One
    plt.style.use('seaborn')
    seaborn.set_context('paper')#, rc={'lines.markeredgewidth': .1})
    fig = plt.figure(figsize=(4,4))
    x = regdf.loc[regdf['ClassL=0.1t0'] == 1,'Diff3MT']
    y = regdf.loc[regdf['ClassL=0.1t0'] == 1,'Diff10YT']
    
    plt.scatter(x, y, color=seaborn.color_palette('deep')[0])
    plt.axhline(0, color="black", linewidth = 0.5, linestyle = '-')
    plt.axvline(0, color="black", linewidth = 0.5, linestyle = '-')
    plt.title('Interest rate behavior on policy days (Narrative one)')
    plt.xlabel('Change in 3-month rate')
    plt.ylabel('Change in 10-year rate')
    plt.xlim(-0.65, 0.2)
    plt.savefig("C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/Latex_MA/Images/ChangePlot02_L0_1.pdf", bbox_inches='tight')
    
    #plot the behavior of interest rates on ppolicy days, Narrative Two
    plt.style.use('seaborn')
    seaborn.set_context('paper')#, rc={'lines.markeredgewidth': .1})
    fig = plt.figure(figsize=(4,4))
    x = regdf.loc[regdf['ClassL=0.1t1'] == 1,'Diff3MT']
    y = regdf.loc[regdf['ClassL=0.1t1'] == 1,'Diff10YT']
    
    plt.scatter(x, y, color=seaborn.color_palette('deep')[0])
    plt.axhline(0, color="black", linewidth = 0.5, linestyle = '-')
    plt.axvline(0, color="black", linewidth = 0.5, linestyle = '-')
    plt.title('Interest rate behavior on policy days (Narrative two)')
    plt.xlabel('Change in 3-month rate')
    plt.ylabel('Change in 10-year rate')
    plt.xlim(-0.65, 0.2)
    plt.savefig("C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/Latex_MA/Images/ChangePlot03_L0_1.pdf", bbox_inches='tight')

       #plot the behavior of interest rates on ppolicy days -- lambda = 0.9 PLSA classification
    # ##################################################################################
    
    #plot the behavior of interest rates on ppolicy days, Narrative One
    plt.style.use('seaborn')
    seaborn.set_context('paper')#, rc={'lines.markeredgewidth': .1})
    fig = plt.figure(figsize=(4,4))
    x = regdf.loc[regdf['ClassL=0.9t0'] == 1,'Diff3MT']
    y = regdf.loc[regdf['ClassL=0.9t0'] == 1,'Diff10YT']
    
    plt.scatter(x, y, color=seaborn.color_palette('deep')[0])
    plt.axhline(0, color="black", linewidth = 0.5, linestyle = '-')
    plt.axvline(0, color="black", linewidth = 0.5, linestyle = '-')
    plt.title('Interest rate behavior on policy days (Narrative one)')
    plt.xlabel('Change in 3-month rate')
    plt.ylabel('Change in 10-year rate')
    plt.xlim(-0.65, 0.2)
    plt.savefig("C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/Latex_MA/Images/ChangePlot02_L0_9.pdf", bbox_inches='tight')
    
    #plot the behavior of interest rates on ppolicy days, Narrative Two
    plt.style.use('seaborn')
    seaborn.set_context('paper')#, rc={'lines.markeredgewidth': .1})
    fig = plt.figure(figsize=(4,4))
    x = regdf.loc[regdf['ClassL=0.9t1'] == 1,'Diff3MT']
    y = regdf.loc[regdf['ClassL=0.9t1'] == 1,'Diff10YT']
    
    plt.scatter(x, y, color=seaborn.color_palette('deep')[0])
    plt.axhline(0, color="black", linewidth = 0.5, linestyle = '-')
    plt.axvline(0, color="black", linewidth = 0.5, linestyle = '-')
    plt.title('Interest rate behavior on policy days (Narrative two)')
    plt.xlabel('Change in 3-month rate')
    plt.ylabel('Change in 10-year rate')
    plt.xlim(-0.65, 0.2)
    plt.savefig("C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/Latex_MA/Images/ChangePlot03_L0_9.pdf", bbox_inches='tight')






    

    #Regression -- classification of PLSA orig, without background topic
    regressionresults = []
    runregdf = pd.DataFrame()
    runregdf['x1'] = regdf['NonPolicyDay']  * regdf['Diff3MT']
    runregdf['x2'] = regdf['ClassL=0.0t0']  * regdf['Diff3MT']
    runregdf['x3'] = regdf['ClassL=0.0t1']  * regdf['Diff3MT']
    for numerator, yieldcurve in enumerate(['Diff6MT', 'Diff1YT', 'Diff3YT', 
                                            'Diff5YT', 'Diff10YT', 'Diff20YT', 'Diff30YT']):
        #run regression with one of the endogenous variables
        runregdf['y'] = regdf[yieldcurve]
        regressionresults.append( smf.ols('y ~  x1 + x2 + x3', data = runregdf).fit() )
        #after running regression, undertake Breusch-Pagan Test to test for heteroskedasticity
        test = sms.het_breuschpagan(regressionresults[numerator].resid, regressionresults[numerator].model.exog)
        if test[1] < 0.05: #check whether there is heteroskedasticity (p-value below 0.05 rejects HO of homoskedasticity)
            print(yieldcurve + 'has heteroskedasticity with p value of ' + str(test[1]))
            regressionresults.pop() #remove regular regression that was already done
            regressionresults.append( smf.ols('y ~  x1 + x2 + x3', data = runregdf).fit(cov_type='HC0') )#run regression with robust sd
        
    regressionresults[5].summary() 
    
    #Regression -- classification of PLSA orig, with background topic, lambda = 0.1
    regressionresults = []
    runregdf = pd.DataFrame()
    runregdf['x1'] = regdf['NonPolicyDay']  * regdf['Diff3MT']
    runregdf['x2'] = regdf['ClassL=0.0t0']  * regdf['Diff3MT']
    runregdf['x3'] = regdf['ClassL=0.0t1']  * regdf['Diff3MT']
    for numerator, yieldcurve in enumerate(['Diff6MT', 'Diff1YT', 'Diff3YT', 
                                            'Diff5YT', 'Diff10YT', 'Diff20YT', 'Diff30YT']):
        #run regression with one of the endogenous variables
        runregdf['y'] = regdf[yieldcurve]
        regressionresults.append( smf.ols('y ~  x1 + x2 + x3', data = runregdf).fit() )
        #after running regression, undertake Breusch-Pagan Test to test for heteroskedasticity
        test = sms.het_breuschpagan(regressionresults[numerator].resid, regressionresults[numerator].model.exog)
        if test[1] < 0.05: #check whether there is heteroskedasticity (p-value below 0.05 rejects HO of homoskedasticity)
            print(yieldcurve + 'has heteroskedasticity with p value of ' + str(test[1]))
            regressionresults.pop() #remove regular regression that was already done
            regressionresults.append( smf.ols('y ~  x1 + x2 + x3', data = runregdf).fit(cov_type='HC0') )#run regression with robust sd
        
    regressionresults[5].summary() 
        
    
  