"""this file contains helper functions to keep our techincal notebook clean"""

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

def top_models(results):
    """this takes in a list of (sorted) cross validation data frames and returns the best result from each"""
    
    model_rows=[]
    for model in results:
        row=model[['params','mean_test_score','std_test_score','mean_fit_time']].iloc[0]
        model_rows.append(row)
    results_df = pd.DataFrame(model_rows)
    results_df.index=['rf_results', 'svm_mc_results', 'svm_ny_results', 'log_results']
    return results_df