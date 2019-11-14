"""this file contains helper functions to keep our techincal notebook clean"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

def top_models(results):
    """this takes in a list of (sorted) cross validation data frames and returns the best result from each"""
    
    model_rows=[]
    for model in results:
        row=model[['params','mean_test_score','std_test_score','mean_fit_time']].iloc[0]
        model_rows.append(row)
    results_df = pd.DataFrame(model_rows)
    results_df.index=['rf_results', 'svm_mc_results', 'svm_ny_results', 'log_results']
    return results_df

def feature_importance(model, X_train, categorical_features, numerical_features):
    """ This returns a dataframe with features and their importances. Input should be a
    fitted random forest model, the training data and the list of categorical features"""
    
    importances=model.feature_importances_
    ohe=OneHotEncoder(drop='first').fit(X_train[categorical_features])
    categorical_names=ohe.get_feature_names()
    feature_names=numerical_features+list(categorical_names)
    importances_df=pd.concat([pd.DataFrame(importances),pd.DataFrame(feature_names)],axis=1)
    importances_df.columns=['Importance','Feature']
    importances_df=importances_df.sort_values(by='Importance',ascending=False)
    return importances_df, ohe