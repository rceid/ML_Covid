# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 10:56:11 2020

@author: diego
"""
import pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss


#%%

filepath = '../data/covid_df.pkl'
df = pd.read_pickle(filepath)
#df['log_cases'] = np.log(df['Confirmed Cases'] + 1)

n_weeks_prediction = 2
df_train, df_test = pipeline.split_and_scale_on_last_weeks(df,
                                                           n_weeks_prediction)
X_train, y_train = pipeline.divide_target_and_features(df_train, 'Confirmed Cases')
X_test, y_test = pipeline.divide_target_and_features(df_test, 'Confirmed Cases')
mlp = MLPClassifier()

mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)

#%%
print(r2_score(y_test, y_pred))
print(r2_score(y_test, y_pred, multioutput='raw_values'))
#pipeline.train_and_evaluate(X_train, y_train, X_test, y_test)

#%%

pipeline.plot_real_vs_prediction(X_test, y_pred, y_test, 'Spain')
pipeline.plot_real_vs_prediction(X_test, y_pred, y_test, 'Portugal')
pipeline.plot_real_vs_prediction(X_test, y_pred, y_test, 'Lebanon')
pipeline.plot_real_vs_prediction(X_test, y_pred, y_test, 'India')
pipeline.plot_real_vs_prediction(X_test, y_pred, y_test, 'United States of America')
pipeline.plot_real_vs_prediction(X_test, y_pred, y_test, 'Peru')
pipeline.plot_real_vs_prediction(X_test, y_pred, y_test, 'Brazil')
pipeline.plot_real_vs_prediction(X_test, y_pred, y_test, 'China')
pipeline.plot_real_vs_prediction(X_test, y_pred, y_test, 'Syria')
pipeline.plot_real_vs_prediction(X_test, y_pred, y_test, 'Laos')
pipeline.plot_real_vs_prediction(X_test, y_pred, y_test, 'Venezuela')
pipeline.plot_real_vs_prediction(X_test, y_pred, y_test, 'South Korea')

#%%

predictions = pipeline.predictions_every_country(X_test.columns[19:169], X_test, y_pred, y_test)
predictions.to_pickle("../data/mlp_predictions.pkl")


