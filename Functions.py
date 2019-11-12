import time
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import matplotlib.cm as cm
import warnings
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

def calculateLiftDiscrete(variableList,Target,variableListColumn):
    df = pd.DataFrame()
    i=0
    for variable in variableList:
        
        total            = variable.shape[0]
        totalFrauds      = Target.value_counts(ascending=False)[1]
        totalCleans      = Target.value_counts(ascending=False)[0]
        table            = variable.value_counts(ascending=False).sort_index()
        propTable        = (variable.value_counts(ascending=False)/variable.value_counts(ascending=False).sum()).sort_index()
        crossTab         = pd.crosstab(variable,Target).sort_values(0,ascending=False).sort_index()
        margineFraudGain = round(crossTab[1]/table,4).sort_index()
        detectionRate    = round(crossTab[1]/totalFrauds,4).sort_index()
        sumCleans        = (table-crossTab[1]).sort_index()
        liftRatio        = (round((crossTab[1]/totalFrauds)/(sumCleans/totalCleans),2)).sort_index()
        tempDF           = pd.DataFrame({
                                       'variableName'     : np.repeat(variableListColumn[i],table.count()),
                                       'category'         : table.index,
                                       'totalTrx'         : table.values,
                                       'totalTrxPerCat'   : propTable.values,
                                       'totalFraud'       : crossTab[1].values,
                                       'totalClean'       : crossTab[0].values,
                                       'margineFraudGain' : margineFraudGain.values,
                                       'detectionRate'    : detectionRate.values,
                                       'lift'             : liftRatio.values

                               })
        df = df.append(tempDF,ignore_index=True)
        i=i+1
    return df
	
def plot_corr_(df,size=10):
    '''Plot a graphical correlation matrix for a dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''
    
    
    import matplotlib.pyplot as plt

    # Compute the correlation matrix for the received dataframe
    corr = df.corr()
    
    # Plot the correlation matrix
    fig, ax = plt.subplots(figsize=(size, size))
    cax = ax.matshow(corr, cmap='coolwarm')
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90);
    plt.yticks(range(len(corr.columns)), corr.columns);
    
    # Add the colorbar legend
    cbar = fig.colorbar(cax, ticks=[-1, 0, 1], aspect=40, shrink=.8)
	


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt