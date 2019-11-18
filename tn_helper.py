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

def feature_importance(model, X_train, categorical_features, numerical_features, drop=True):
    """ This returns a dataframe with features and their importances. Input should be a
    fitted random forest model, the training data and the list of categorical features"""
    
    importances=model.feature_importances_
    if drop:
        ohe=OneHotEncoder(drop='first').fit(X_train[categorical_features])
    else:
        ohe=OneHotEncoder(drop=None).fit(X_train[categorical_features])
    categorical_names=ohe.get_feature_names()
    feature_names=numerical_features+list(categorical_names)
    importances_df=pd.concat([pd.DataFrame(importances),pd.DataFrame(feature_names)],axis=1)
    importances_df.columns=['Importance','Feature']
    importances_df=importances_df.sort_values(by='Importance',ascending=False)
    return importances_df

def quick_f_i_plot():
    """ """
    preprocess_no_drop = make_column_transformer(
    (StandardScaler(), numerical_features),
    (OneHotEncoder(drop = None), categorical_features)
    )

    random_forest_no_drop=RandomForestClassifier(
        max_depth=30, max_features='sqrt', min_samples_leaf=10, n_estimators=300, n_jobs=-1,oob_score=True)

    t_X_train_no_drop=preprocess_no_drop.fit_transform(X_train)
    t_X_test_no_drop=preprocess_no_drop.fit_transform(X_test)

    random_forest_no_drop.fit(t_X_train_no_drop, y_train)

    print('Training Accuracy: ', random_forest_no_drop.score(t_X_train_no_drop,y_train))

    print('Testing Accuracy: ', random_forest_no_drop.score(t_X_test_no_drop,y_test))

    importances_df_no_drop=tn.feature_importance(random_forest_no_drop, X_train, categorical_features, numerical_features, drop=False)

    decoded_imports2=[]
    for idx, feature in enumerate(categorical_features):
        selection=importances_df_no_drop.loc[importances_df_no_drop.Feature.str.contains(f'x{idx}_')]
        tot_import=selection.Importance.sum()
        decoded_imports2.append((tot_import, feature))

    decode_df2=pd.DataFrame(decoded_imports2, columns=['Importance','Feature'])
    full_decoded_df2=pd.concat([importances_df_no_drop[0:15], decode_df2], axis=0)
    full_decoded_df2.sort_values('Importance', inplace=True, ascending=False)