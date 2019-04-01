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
from decimal import getcontext, Decimal




if __name__ == '__main__':
    
    #Reading in data
    #===============
    path ="C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/MA_FinancialData" #absolute path to the txt files
    os.chdir(path) #setting working directory
    
    #read in the FED data on target rate
    financedf = pd.read_csv('financeData.csv', sep = ',')
    
    path ="C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/Code/Output" #absolute path to the txt files
    os.chdir(path) #setting working directory
    
    #read in classification of policy days, lambda = 0.0-0.9
    #classL0_9 = pd.read_csv('AdjustmentsClassifiedPLSAbgLamb_0_9.csv', sep = ',')
    classL0_9 = pd.read_csv('AdjustmentsClassifiedPLSAbg_modLamb_0_9.csv', sep = ',')
    
    
    #Correct time frame for financial data
    #turn date from string into datetime object
    for i in range(len(financedf)): #replace date string with date format
        financedf.iloc[i,0] = datetime.datetime.strptime(financedf.iloc[i,0], '%Y-%m-%d')
    
    for l in range(len(classL0_9)): #replace date string with date format
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
    for lamb in ['L=0.0','L=0.1', 'L=0.2', 'L=0.3', 'L=0.4', 'L=0.5', 'L=0.6', 
                 'L=0.7', 'L=0.8', 'L=0.9']:
        for topic in ['t0', 't1']:
            regdf['Class'+lamb+topic] = 0
    
    for i in range(len(regdf['Date'])):
        day = regdf.loc[i, 'Date']
        if day in list(classL0_9['Date']):
            rowindex = classL0_9[classL0_9['Date'] == day].index.values.astype(int)[0] #find row of date in classification df
            if classL0_9.loc[rowindex, 'ClassificationLamb_0_0'] == 0:
                regdf.loc[i,'ClassL=0.0t0'] = 1
            if classL0_9.loc[rowindex, 'ClassificationLamb_0_0'] == 1:
                regdf.loc[i,'ClassL=0.0t1'] = 1
            if classL0_9.loc[rowindex, 'ClassificationLamb_0_1'] == 0:
                regdf.loc[i,'ClassL=0.1t0'] = 1
            if classL0_9.loc[rowindex, 'ClassificationLamb_0_1'] == 1:
                regdf.loc[i,'ClassL=0.1t1'] = 1
                
            if classL0_9.loc[rowindex, 'ClassificationLamb_0_2'] == 0:
                regdf.loc[i,'ClassL=0.2t0'] = 1
            if classL0_9.loc[rowindex, 'ClassificationLamb_0_2'] == 1:
                regdf.loc[i,'ClassL=0.2t1'] = 1
            if classL0_9.loc[rowindex, 'ClassificationLamb_0_3'] == 0:
                regdf.loc[i,'ClassL=0.3t0'] = 1
            if classL0_9.loc[rowindex, 'ClassificationLamb_0_3'] == 1:
                regdf.loc[i,'ClassL=0.3t1'] = 1
            if classL0_9.loc[rowindex, 'ClassificationLamb_0_4'] == 0:
                regdf.loc[i,'ClassL=0.4t0'] = 1
            if classL0_9.loc[rowindex, 'ClassificationLamb_0_4'] == 1:
                regdf.loc[i,'ClassL=0.4t1'] = 1
            if classL0_9.loc[rowindex, 'ClassificationLamb_0_5'] == 0:
                regdf.loc[i,'ClassL=0.5t0'] = 1
            if classL0_9.loc[rowindex, 'ClassificationLamb_0_5'] == 1:
                regdf.loc[i,'ClassL=0.5t1'] = 1
            if classL0_9.loc[rowindex, 'ClassificationLamb_0_6'] == 0:
                regdf.loc[i,'ClassL=0.6t0'] = 1
            if classL0_9.loc[rowindex, 'ClassificationLamb_0_6'] == 1:
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
    + regdf['ClassL=0.9t0'] + regdf['ClassL=0.9t1']) != 10*len(classL0_9):
        raise Exception("The data frame for regression is not constructed correctly, check the assignment of topics to policy days (Lambda = 0.0)!")
    if sum(regdf['NonPolicyDay'] + regdf['ClassL=0.1t0'] + regdf['ClassL=0.1t1']) != len(regdf):
        raise Exception("The data frame for regression is not constructed correctly, check the assignment of topics to policy days (Lambda = 0.1)!")
    
    
    #plot the behavior of interest rates on ppolicy days -- original PLSA classification
    # ##################################################################################
    
    #plot the behavior of interest rates on ppolicy days, Narrative One & Two
    plt.style.use('seaborn')
    seaborn.set_context('paper')#, rc={'lines.markeredgewidth': .1})
    fig = plt.figure(figsize=(4,4))
   
    x = regdf.loc[regdf['ClassL=0.0t1'] == 1,'Diff3MT']
    y = regdf.loc[regdf['ClassL=0.0t1'] == 1,'Diff10YT']
    plt.scatter(x, y, color=seaborn.color_palette('deep')[1], alpha = 1, label='Narrative Two')
    
    x = regdf.loc[regdf['ClassL=0.0t0'] == 1,'Diff3MT']
    y = regdf.loc[regdf['ClassL=0.0t0'] == 1,'Diff10YT']
    plt.scatter(x, y, color=seaborn.color_palette('deep')[0], alpha = 1, label='Narrative One')
    
    plt.axhline(0, color="black", linewidth = 0.5, linestyle = '-')
    plt.axvline(0, color="black", linewidth = 0.5, linestyle = '-')
    plt.title('Interest rate behavior on policy days')
    plt.xlabel('Change in 3-month rate')
    plt.ylabel('Change in 10-year rate')
    plt.xlim(-0.65, 0.2)
    plt.legend()
    #plt.savefig("C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/Latex_MA/Images/ChangePlot_L0_0.pdf", bbox_inches='tight')
    plt.savefig("C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/Latex_MA/Images/ChangePlotmod_L0_0.pdf", bbox_inches='tight')
 
           #plot the behavior of interest rates on ppolicy days -- lambda = 0.1 PLSA classification
    # ##################################################################################
    
    #plot the behavior of interest rates on ppolicy days, Narrative One
    plt.style.use('seaborn')
    seaborn.set_context('paper')#, rc={'lines.markeredgewidth': .1})
    fig = plt.figure(figsize=(4,4))
    
    x = regdf.loc[regdf['ClassL=0.1t1'] == 1,'Diff3MT']
    y = regdf.loc[regdf['ClassL=0.1t1'] == 1,'Diff10YT']
    plt.scatter(x, y, color=seaborn.color_palette('deep')[1], alpha = 1, label='Narrative Two')
    
    x = regdf.loc[regdf['ClassL=0.1t0'] == 1,'Diff3MT']
    y = regdf.loc[regdf['ClassL=0.1t0'] == 1,'Diff10YT']
    plt.scatter(x, y, color=seaborn.color_palette('deep')[0], alpha = 1, label='Narrative One')
   
    plt.axhline(0, color="black", linewidth = 0.5, linestyle = '-')
    plt.axvline(0, color="black", linewidth = 0.5, linestyle = '-')
    plt.title('Interest rate behavior on policy days')
    plt.xlabel('Change in 3-month rate')
    plt.ylabel('Change in 10-year rate')
    plt.xlim(-0.65, 0.2)
    plt.legend()
    #plt.savefig("C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/Latex_MA/Images/ChangePlot_L0_1.pdf", bbox_inches='tight')
    plt.savefig("C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/Latex_MA/Images/ChangePlotmod_L0_1.pdf", bbox_inches='tight')


       #plot the behavior of interest rates on ppolicy days -- lambda = 0.9 PLSA classification
    # ##################################################################################
    
    #plot the behavior of interest rates on ppolicy days, Narrative One
    plt.style.use('seaborn')
    seaborn.set_context('paper')#, rc={'lines.markeredgewidth': .1})
    fig = plt.figure(figsize=(4,4))
    
    x = regdf.loc[regdf['ClassL=0.9t1'] == 1,'Diff3MT']
    y = regdf.loc[regdf['ClassL=0.9t1'] == 1,'Diff10YT']
    plt.scatter(x, y, color=seaborn.color_palette('deep')[1], alpha = 1, label='Narrative Two')
    
    x = regdf.loc[regdf['ClassL=0.9t0'] == 1,'Diff3MT']
    y = regdf.loc[regdf['ClassL=0.9t0'] == 1,'Diff10YT']
    plt.scatter(x, y, color=seaborn.color_palette('deep')[0], alpha = 1, label='Narrative One')
   
    plt.axhline(0, color="black", linewidth = 0.5, linestyle = '-')
    plt.axvline(0, color="black", linewidth = 0.5, linestyle = '-')
    plt.title('Interest rate behavior on policy days')
    plt.xlabel('Change in 3-month rate')
    plt.ylabel('Change in 10-year rate')
    plt.xlim(-0.65, 0.2)
    plt.legend()
    #plt.savefig("C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/Latex_MA/Images/ChangePlot_L0_9.pdf", bbox_inches='tight')
    plt.savefig("C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/Latex_MA/Images/ChangePlotmod_L0_9.pdf", bbox_inches='tight')


    #Regression -- run all regressions and print results in output that can be used for Latex table
    ###############################################################################################
    Classifiers = []
    for TS in range(12, len(list(regdf)), 2):
        tup = list(regdf)[TS:TS+2]
        tup = tuple(tup)
        Classifiers.append(tup)
    
    for classification in Classifiers:

        regressionresults = []
        runregdf = pd.DataFrame()
        runregdf['x1'] = regdf['NonPolicyDay']  * regdf['Diff3MT']
        runregdf['x2'] = regdf[classification[0]]  * regdf['Diff3MT'] #narrative one
        runregdf['x3'] = regdf[classification[1]]  * regdf['Diff3MT'] #narrative two
        for numerator, yieldcurve in enumerate(['Diff6MT', 'Diff1YT', 'Diff3YT', 
                                                'Diff5YT', 'Diff10YT', 'Diff20YT', 'Diff30YT']):
            #run regression with one of the endogenous variables
            runregdf['y'] = regdf[yieldcurve]
            regressionresults.append( smf.ols('y ~  x1 + x2 + x3', data = runregdf).fit() )
            #after running regression, undertake Breusch-Pagan Test to test for heteroskedasticity
# =============================================================================
#             test = sms.het_breuschpagan(regressionresults[numerator].resid, regressionresults[numerator].model.exog)
#             if test[1] < 0.05: #check whether there is heteroskedasticity (p-value below 0.05 rejects HO of homoskedasticity)
#                 print(yieldcurve + ' has heteroskedasticity with p value of ' + str(test[1]))
# =============================================================================
            regressionresults.pop() #remove regular regression that was already done
            regressionresults.append( smf.ols('y ~  x1 + x2 + x3', data = runregdf).fit(cov_type='HC0') )#run regression with robust sd
            
        
        #print the regression values for a Latex table
        #make graph of betas and confidence intervals
        RegressionValues = {}
        
        #add coefficients to dictionary
        para = -1
        for element in ['$\\alpha_n$', '$\\beta_n^{NP}$', '$\\beta_n^{N1}$', '$\\beta_n^{N2}$']:
            para += 1
            RegressionValues[element] = []
            for i in range(7):
                RegressionValues[element].append("%.2f" % regressionresults[i].params[para]) #adding all the intercepts
                if regressionresults[i].pvalues[para] <= 0.05 and regressionresults[i].pvalues[para] > 0.01:
                    RegressionValues[element].append('*')
                if regressionresults[i].pvalues[para] <= 0.01:
                    RegressionValues[element].append('**')
                RegressionValues[element].append(' & ')
            RegressionValues[element].pop() #removing the last '&'
            RegressionValues[element].append("\\\\")
        
        #add standard errors to dictionary
        para = -1
        for element in ['std0', 'std1', 'std2', 'std3']:
            para += 1
            RegressionValues[element] = []
            for i in range(7):
                RegressionValues[element].append("(" + "%.2f" % regressionresults[i].bse[para] + ")") #adding all the std in brakets
                RegressionValues[element].append(' & ')
            RegressionValues[element].pop() #removing the last '&'
            RegressionValues[element].append("\\\\")
        
        #add R2 to dictionary
        RegressionValues['R2'] = []
        for i in range(7):
            RegressionValues['R2'].append("%.2f" % regressionresults[i].rsquared)
            RegressionValues['R2'].append(' & ')
        RegressionValues['R2'].pop() #removing the last '&'
        RegressionValues['R2'].append("\\\\")
        
        #add equality test to table
        key = '$\\beta_n^{N1} = \\beta_n^{N2}$'
        RegressionValues[key] = []
        hypothesis = 'x2 = x3'
        for i in range(7):
            t_test = regressionresults[i].t_test(hypothesis)
            #print(t_test)
            RegressionValues[key].append("%.2f" % t_test.effect)
            if t_test.pvalue <= 0.05 and t_test.pvalue > 0.01:
                  RegressionValues[key].append('*')
            if t_test.pvalue <= 0.01:  
                RegressionValues[key].append('**')
            RegressionValues[key].append(' & ')
        RegressionValues[key].pop() #removing the last '&'
        RegressionValues[key].append("\\\\")

        
        print('######################################')
        print("Summary of regression with " + classification[0][:-2])
        print('######################################')
        print ('$\\alpha_n$'  + ' & ' + ''.join(RegressionValues['$\\alpha_n$']))
        print (" & " + ''.join(RegressionValues["std0"]) )
        print ('$\\beta_n^{NP}$'  + ' & ' + ''.join(RegressionValues['$\\beta_n^{NP}$']))
        print (" & " + ''.join(RegressionValues["std1"]) )
        print ('$\\beta_n^{N1}$'  + ' & ' + ''.join(RegressionValues['$\\beta_n^{N1}$']))
        print (" & " + ''.join(RegressionValues["std2"]) )
        print ('$\\beta_n^{N2}$'  + ' & ' + ''.join(RegressionValues['$\\beta_n^{N2}$']))
        print (" & " + ''.join(RegressionValues["std3"]) )
        print ("$R^2$" + " & " + ''.join(RegressionValues["R2"]) )
        print ('$\\beta_n^{N1} = \\beta_n^{N2}$' + " & " + ''.join(RegressionValues['$\\beta_n^{N1} = \\beta_n^{N2}$']) )
        
        
        
        #build a dictionary for graph
        GraphDic = {}
        
        #add coefficients to dictionary
        para = 1
        for element in ['beta1', 'beta2']:
            para += 1
            GraphDic[element] = []
            for i in range(7):
                GraphDic[element].append(regressionresults[i].params[para]) 
        
        #add confidence intervals to dictionary
        para = 1
        for element in ['b1conflow', 'b2conflow']:
            para += 1
            GraphDic[element] = []
            for i in range(7):
                GraphDic[element].append(regressionresults[i].conf_int()[0][para]) 
                
        para = 1
        for element in ['b1conhigh', 'b2conhigh']:
            para += 1
            GraphDic[element] = []
            for i in range(7):
                GraphDic[element].append(regressionresults[i].conf_int()[1][para]) 
              
        #plot the behavior of interest rates on ppolicy days, Narrative One
        plt.style.use('seaborn')
        seaborn.set_context('paper')#, rc={'lines.markeredgewidth': .1})
        fig, ax = plt.subplots(figsize=(7,4))
        xlabel = ['6MT', '1YT', '3YT', '5YT', '10YT', '20YT', '30YT']
        
        ax.plot(GraphDic['beta1'], label='Beta of Narrative 1', 
             linewidth=1.5, color=seaborn.color_palette('deep')[0])
        ax.plot(GraphDic['b1conflow'], 
             linewidth=.75, linestyle = "--", color=seaborn.color_palette('deep')[0])
        ax.plot(GraphDic['b1conhigh'], 
             linewidth=.75, linestyle = "--", color=seaborn.color_palette('deep')[0])
        
        ax.fill_between(range(len(xlabel)), GraphDic['b1conflow'], GraphDic['b1conhigh'], alpha=0.2)
         
        ax.plot(GraphDic['beta2'], label='Beta of Narrative 2', 
             linewidth=1.5, color=seaborn.color_palette('deep')[1])
        ax.plot(GraphDic['b2conflow'], 
             linewidth=.75, linestyle = "--", color=seaborn.color_palette('deep')[1])
        ax.plot(GraphDic['b2conhigh'], 
             linewidth=.75, linestyle = "--", color=seaborn.color_palette('deep')[1])
        
        ax.fill_between(range(len(xlabel)), GraphDic['b2conflow'], GraphDic['b2conhigh'], alpha=0.2)
        
        ax.set_xlim(xmin=0, xmax=len(xlabel)-1)
        plt.xticks(list(range(len(xlabel))), xlabel)
        #plt.title("Regression coefficients for narrative one and two")
        plt.xlabel("Maturities of interest rates")
        ax.legend(loc='upper right', frameon=False)
        #plt.savefig("C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/Latex_MA/Images/betasLamb" + str(classification[0][7]) + "_" + str(classification[0][9]) + ".pdf", bbox_inches='tight')
        plt.savefig("C:/Users/corin/Documents/Uni/M.A.HSG/MA_Arbeit/MasterThesis_NarrativesInFinance/Latex_MA/Images/betasLambMod" + str(classification[0][7]) + "_" + str(classification[0][9]) + ".pdf", bbox_inches='tight')

       
        sum(regdf['ClassL=0.9t1'])
        
    
  