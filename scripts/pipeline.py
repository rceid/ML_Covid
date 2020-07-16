# -*- coding: utf-8 -*-
"""
Created on Wed May  6 16:55:31 2020

@author: diego - piyush - raymond
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import datetime
from datetime import timedelta
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import math


def metrics(y_pred, y_test, x_train, y_train, model, output=True):
    '''
    Returns and prints 5 classic metrics for a machine learning model

    Parameters
    ----------
    y_pred : TYPE
        DESCRIPTION.
    y_test : TYPE
        DESCRIPTION.
    x_train : TYPE
        DESCRIPTION.
    y_train : TYPE
        DESCRIPTION.
    model : TYPE
        DESCRIPTION.
    output : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    floats.

    '''

    bias = mean_squared_error(y_train, model.predict(x_train))
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    rss = np.sum((y_pred - y_test) ** 2)
    variance = model.score(x_train, y_train)
    r2_s = r2_score(y_test, y_pred)

    if output:
        print("Bias: %.2f" % bias)
        print("Root Mean squared error: %.2f" % rmse)
        print("RSS: %.2f" % rss)
        print('Variance score: %.2f\n' % variance)
        print('R2 score: %.2f\n' % r2_s)

    return(bias, rmse, rss, variance, r2_s)


def get_most_relevant_features(df, model, number_of_features):
    '''
    Returns a sorted dataframe of the most relevant features.
    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    model : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    features = pd.DataFrame(columns=['Feature', 'Coefficient', 'abs'])
    features = features.append({'Feature': 'Intercept', 'Coefficient':
                               model.intercept_, 'abs': abs(model.intercept_)},
                               ignore_index=True)
    for i in range(len(df.columns)):
        features = features.append({'Feature': df.columns[i],
                                    'Coefficient': model.coef_[i],
                                    'abs': abs(model.coef_[i])},
                                    ignore_index=True)

    features = features.sort_values(by=['abs'], ascending=False)
    features = features.drop(['abs'], axis=1)

    labels = features['Feature']
    feats = features['Coefficient']
    indices = np.argsort(features['Coefficient'])[::-1]
    names = [labels[i] for i in indices]
    colors = ['r' if coef <0 else 'royalblue' for coef in feats[:5]]
    plt.figure()
    print(features)
    plt.bar(labels[:5], abs(feats[:5]), color=colors)
    plt.xticks(labels[:5], names[:5], rotation=90)
    plt.show()
    return features[:number_of_features]


def read_and_process_data(filepath):
    '''
    Reads data from a filepath, processes it, and splits in train and test sets

    Returns
    -------
    None.

    '''
    split_date = datetime.date(2020, 5, 1)
    df_train, df_test = read_split_and_scale(filepath, split_date)
    df_train['Country'].astype('category')
    df_test['Country'].astype('category')
    df_train = df_train.drop(['Date'], axis=1)
    df_test = df_test.drop(['Date'], axis=1)
    df_train = pd.get_dummies(df_train)
    df_test = pd.get_dummies(df_test)
    df_train = remove_countries_not_in_test_set(df_train, df_test)
    return df_train, df_test


def read_split_and_scale(filepath, split_date):
    '''
    Reads dataframe from filepath, splits data on training/testing data, scales
    it, merges scalable and non scalable columns, and returns both training and
    testinf dataframes. At first, since the scaling process forms a new
    dataframe, we create an index column so we can merge back afterwards.

    Returns
    -------
    df_train, df_test.

    '''
    df = pd.read_pickle(filepath)
    df_training, df_test = split_on_date(df, split_date)
    scalable_train, non_scalable_train = split_scalable_columns(df_training)
    scalable_test, non_scalable_test = split_scalable_columns(df_test)
    scaled_train, scaled_test = scale_df(scalable_train, scalable_test)

    scaled_train = scaled_train.reset_index()
    scaled_train = scaled_train.drop(['index'], axis=1)
    non_scalable_train = non_scalable_train.reset_index()
    non_scalable_train = non_scalable_train.drop(['index'], axis=1)
    scaled_test = scaled_test.reset_index()
    scaled_test = scaled_test.drop(['index'], axis=1)
    non_scalable_test = non_scalable_test.reset_index()
    non_scalable_test = non_scalable_test.drop(['index'], axis=1)
    df_train = scaled_train.join(non_scalable_train)
    df_test = scaled_test.join(non_scalable_test)

    return df_train, df_test

def split_and_scale_on_last_weeks(df, n_weeks_prediction):
    '''


    Parameters
    ----------
    filepath : TYPE
        DESCRIPTION.
    n_weeks : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    split_date = df['Date'].max() - timedelta(days=n_weeks_prediction*7)
    df_training, df_test = split_on_date(df, split_date)
    scalable_train, non_scalable_train = split_scalable_columns(df_training)
    scalable_test, non_scalable_test = split_scalable_columns(df_test)
    scaled_train, scaled_test = scale_df(scalable_train, scalable_test)

    scaled_train = scaled_train.reset_index()
    scaled_train = scaled_train.drop(['index'], axis=1)
    non_scalable_train = non_scalable_train.reset_index()
    non_scalable_train = non_scalable_train.drop(['index'], axis=1)
    scaled_test = scaled_test.reset_index()
    scaled_test = scaled_test.drop(['index'], axis=1)
    non_scalable_test = non_scalable_test.reset_index()
    non_scalable_test = non_scalable_test.drop(['index'], axis=1)
    df_train = scaled_train.join(non_scalable_train)
    df_test = scaled_test.join(non_scalable_test)

    df_train = make_category_types(df_train)
    df_test = make_category_types(df_test)
    df_train = df_train.drop(['Date'], axis=1)
    df_test = df_test.drop(['Date'], axis=1)
    df_train = pd.get_dummies(df_train)
    df_test = pd.get_dummies(df_test)
    df_train = remove_countries_not_in_test_set(df_train, df_test)

    return df_train, df_test


def make_category_types(df):
    '''
    Transforms the variables that should be one hot encoded to categorical
    type

    Parameters
    ----------
    df : Dataframe with features.

    Returns
    -------
    df : Dataframe.

    '''
    X_vars = ['Country', 'C1_School closing', 'C2_Workplace closing',
              'C3_Cancel public events', 'C4_Restrictions on gatherings',
              'C5_Close public transport', 'C6_Stay at home requirements',
              'C7_Restrictions on internal movement',
              'C8_International travel controls', 'E1_Income support',
              'E2_Debt/contract relief', 'H1_Public information campaigns',
              'H2_Testing policy', 'H3_Contact tracing']

    for var in X_vars:
        df[var] = df[var].astype('category')
    return df


def cut_df_on_weeks(df, n_weeks):
    '''


    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    n_weeks : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    cut_date = df['Date'].max() - timedelta(days=n_weeks*7)
    print("Cutting dataframe on date: " + str(cut_date))
    df = df.loc[df['Date'] <= cut_date]
    return df


def split_scalable_columns(df):
    '''
    Splits df into scalable columns and non-scalable columns

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    scalable_vars : TYPE
        DESCRIPTION.
    non_scalable_vars : TYPE
        DESCRIPTION.

    '''

    non_scalable = ['Country', 'C1_School closing', 'C2_Workplace closing',
                    'C3_Cancel public events', 'C4_Restrictions on gatherings',
                    'C5_Close public transport', 'C6_Stay at home requirements'
                    , 'C7_Restrictions on internal movement',
                    'C8_International travel controls', 'E1_Income support',
                    'E2_Debt/contract relief', 'H1_Public information campaigns',
                    'H2_Testing policy', 'H3_Contact tracing', 'Date',
                    'Day Count', 'Days Elapsed Since First Case',
                    'Confirmed Cases', 'Deaths', 'Recovered',
                    'Daily New Cases', 'Daily Deaths', 'log_cases']
    non_scalable_vars = df[non_scalable]
    lst = []
    for column in df.columns:
        if column not in non_scalable_vars.columns:
            lst.append(column)
    scalable_vars = df[lst]
    return scalable_vars, non_scalable_vars


def split_on_date(df, split_date):
    '''


    Parameters
    ----------
    split_date : Datetime date.

    Returns
    -------
    train_df, test_df.

    '''
    df_training = df.loc[df['Date'] <= split_date]
    df_test = df.loc[df['Date'] > split_date]

    return df_training, df_test


def scale_df(scalable_train, scalable_test):
    '''
    Scales df, assumes all columns are scalable

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    ''' '''
    train_index = scalable_train['index']
    scalable_train = scalable_train.drop(['index'], axis=1)
    test_index = scalable_test['index']
    scalable_test = scalable_test.drop(['index'], axis=1)
    '''
    scaler = preprocessing.StandardScaler().fit(scalable_train)
    scaled_train = pd.DataFrame(scaler.transform(scalable_train),
                                columns=scalable_train.columns.values)
    scaled_test = pd.DataFrame(scaler.transform(scalable_test),
                               columns=scalable_test.columns)
    '''
    scaled_train['index'] = train_index.astype(int)
    scaled_test['index'] = test_index.astype(int)
    '''
    return scaled_train, scaled_test


def sanity_check(train_df, test_df):

    # Sort features alphabetically
    train_df = train_df.reindex(sorted(train_df.columns), axis=1)
    test_df = test_df.reindex(sorted(test_df.columns), axis=1)

    # Check that they have the same features
    if (train_df.columns == test_df.columns).all():
        print("Success: Features match")
    else:
        print("Data not clean yet, one or more features do not match")

    # Check that no NAs remain
    condition_1 = not train_df.isna().sum().astype(bool).any()
    condition_2 = not test_df.isna().sum().astype(bool).any()
    if condition_1 and condition_2:
        print("Success: No NAs remain")
    else:
        print("Failure: Data is not clean yet, NAs remaining")


def remove_countries_not_in_test_set(df_train, df_test):
    '''
    Removes variables not present in test set

    Parameters
    ----------
    train_df : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    extra_countries = [col for col in df_train.columns if col not in
                       df_test.columns]
    df_train = df_train.drop(extra_countries, axis=1)
    return df_train


def divide_target_and_features(df, target):
    '''


    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    target : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    y = df[target]
    outcome_vars = ['Confirmed Cases', 'Deaths', 'Recovered',
                    'Daily New Cases', 'Daily Deaths', 'log_cases']
    x = df.drop(outcome_vars, axis=1)
    return x, y


def divide_target_and_one_feature(df, target):
    '''


    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    target : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    y = df[target]
    x = df.drop([target], axis=1)
    return x, y


def train_and_evaluate(x_train, y_train, x_test, y_test):
    '''


    Parameters
    ----------
    x_train : TYPE
        DESCRIPTION.
    y_train : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    ls = linear_model.Lasso(alpha=0.5)
    rg = linear_model.Ridge(alpha=0.5)
    lreg = linear_model.LinearRegression()
    ev = {}
    models = [(ls, 'Lasso'),
              (rg, 'Ridge'),
              (lreg, 'Linear Regression')]

    for m in models:
        (model, name) = m
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print('{}\n{}\n'.format(name + ': Features with highest magnitude\
                                coefficients in absolute value',
                                get_most_relevant_features(x_train,
                                                           model, 10)))
        ev[name] = metrics(np.exp(y_pred), np.exp(y_test), x_train, np.exp(y_train), model)

    return ev


def train_and_evaluate_w_grid(x_train, y_train, x_test, y_test):
    '''


    Parameters
    ----------
    x_train : TYPE
        DESCRIPTION.
    y_train : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    ls = GridSearchCV(linear_model.Lasso(), {'alpha': [0.1, 0.2, 0.3, 0.4,
                                                       0.5]})
    rg = GridSearchCV(linear_model.Ridge(), {'alpha': [0.1, 0.2, 0.3, 0.4,
                                                       0.5]})
    ev = {}
    models = [(ls, 'Lasso'),
              (rg, 'Ridge')]

    for m in models:
        (model, name) = m
        model.fit(x_train, y_train)
        ev[name] = model.cv_results_
        print(ev[name])

    return ev


def plot_real_vs_prediction(X_test, y_pred, y_test, country_name):
    '''


    Parameters
    ----------
    X_test : TYPE
        DESCRIPTION.
    y_pred : TYPE
        DESCRIPTION.
    y_test : TYPE
        DESCRIPTION.
    country_name : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    country = 'Country_' + country_name
    x = X_test['Day Count'][X_test[country] == 1].apply(
        lambda x: timedelta(x) + datetime.date(2020, 1, 1))

    plt.plot(x, y_pred[X_test[country] == 1], marker='o', markerfacecolor=
             'blue', markersize=12, color='skyblue', linewidth=4, label='Real')
    plt.plot(x, y_test[X_test[country] == 1], marker='', color='olive',
             linewidth=2, label='Prediction')
    plt.legend()


def predictions_every_country(country_list, X_test, y_pred, y_test):
    '''


    Parameters
    ----------
    country_list : TYPE
        DESCRIPTION.
    X_test : TYPE
        DESCRIPTION.
    y_pred : TYPE
        DESCRIPTION.
    y_test : TYPE
        DESCRIPTION.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    '''
    df = pd.DataFrame()
    df_final = pd.DataFrame()
    dates = X_test['Day Count'][X_test['Country_United States of America']
                                == 1].apply(lambda x: timedelta(x) +
                                            datetime.date(2020, 1, 1))
    df['date'] = dates
    df = df.set_index('date')

    for country_var in country_list:
        dates = X_test['Day Count'][X_test[country_var] == 1].\
                apply(lambda x: timedelta(x) + datetime.date(2020, 1, 1))
        df_aux = pd.DataFrame()
        df_aux['date'] = dates
        df_aux[country_var[8:] + ' real'] = y_test[X_test[country_var] == 1]
        df_aux[country_var[8:] + ' prediction'] = y_pred[X_test[country_var]
                                                         == 1]
        df_aux = df_aux.set_index('date')
        df_final = df_final.join(df_aux.copy(), how='outer')

    return df_final



def get_first_case(jhu_df):
    '''
    From John Hopkins data, gets the date of first case for each country and
    returns as a two column dataframe
    '''
    sub = jhu_df[jhu_df['Confirmed'] > 0.0]
    first_case = sub.groupby('Country')['Date'].min().reset_index().\
                 sort_values(by=['Country'])
    return first_case


def read_and_clean_oxford_data():
    '''
    Reads Oxford data from git url

    Parameters
    ----------
    url : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    oxford_policy = pd.read_csv(OXFORD_URL)
    cols_to_keep = [col for col in list(oxford_policy.columns) if\
                    'Flag' not in col]
    oxford_policy = oxford_policy[cols_to_keep]
    oxford_policy['Date'] = pd.to_datetime(oxford_policy['Date'].astype(str),\
                                           format='%Y%m%d')
    oxford_policy['Date'] = pd.to_datetime(oxford_policy['Date'],\
                                           infer_datetime_format=True)
    oxford_policy['Date'] = oxford_policy['Date'].dt.date
    oxford_policy.rename(columns={'CountryName': 'Country'}, inplace=True)
    oxford_policy['Country'] = oxford_policy['Country'].replace(countries_names)
    return oxford_policy

def read_wb_data(data_dir):
    '''
    Reads and returns world bank data
    '''
    pop_old = pd.read_excel(data_dir+'popabove65.xls', sheet_name='Data',\
                            header=3, usecols = ["Country Name", "Country Code", "2018"])
    hbeds = pd.read_excel(data_dir+'hospbeds.xls', sheet_name='Data',\
                          header=3)
    yrs = hbeds[hbeds.columns[4:]]
    yrs = yrs.ffill(axis=1)['2019']
    wb_data = pd.concat([hbeds['Country Name'], yrs, pop_old[['Country Code', '2018']]], axis=1)
    wb_data.rename(columns={'2019': 'Hospital Beds/1k', 'Country Name':\
                            'Country', '2018': 'Share Pop 65+'}, inplace=True)
    return wb_data


def side_by_side(full_df, country, target, models, *predictions,
                 save_output=False):
    '''
    Plots two different plots of models' predictions vs real trends side by side.
    --first *predictions arg must be the one to be logged (LinReg)
    Inputs:
        full_df: (Pandas df) The full cleaned dataset
        country: (string) Country to examine
        target: (string) the outcome variable of the model
        models: (list) model names as strings
        *predictions: (tuple of Pandas df) collection of model prediction data
        save_output: (boolean) switch to save image output
    Output:
        File if save_output=True
        plots figure
    '''
    np.seterr(all='ignore')
    style.use('seaborn')
    date = predictions[0].index.min()
    pre = full_df[(full_df['Country'] == country) & (full_df['Date']
                                                     <= date)][[target, 'Date']]
    real, predict = (country + ' real', country + ' prediction')
    day = pre['Date'].max() + dt.timedelta(days=-1)
    fig, axes = plt.subplots(1, 2, figsize=(20,5))
    title = ' Prediction: {} in {}'.format(target, country)
    post_trends = []
    y_max = 0

    for pred, model in zip(predictions, models):
        post = pred[[real, predict]]
        row = pre[pre['Date'] == day][[target, 'Date']]
        row.set_index('Date', inplace=True)
        val = row[target]
        if model == 'Linear Regression':
            row[real], row[predict] = (np.log(val), np.log(val))
            post = np.exp(row.append(post[[real, predict]]))
        else:
            row[real], row[predict] = (val, val)
            post = row.append(post[[real, predict]])
        post_trends.append(post)
    iterable = zip(models, predictions, axes, post_trends)
    for model, output, axis, trend in iterable:
        sub_title = model + title
        axis.title.set_text(sub_title)
        axis.axvline(x=date, ls=':', c='gray', label = str(date))
        g = sns.lineplot(x=trend.index, y=trend[real], ax=axis, marker='X', color='darkorange')
        g = sns.lineplot(x=trend.index, y=trend[predict], ax=axis, marker='X', color='g')
        g = sns.lineplot(x=pre['Date'], y=pre[target], ax=axis, color='royalblue')
        axis.legend(('Prediction frontier\n {}'.format(date), 'Real', 'Predicted', 'Trend'), prop={'size': 12})
        plt.ylabel(target)
        if output[[real, predict]].dropna().values.max() > y_max:
            y_max = output[[real, predict]].dropna().values.max()
    plt.ylim(0, y_max + y_max*.15)
    if save_output:
        file_name = '{} {} {} comparison.png'.format(country, *models)
        plt.savefig('.\\..\\visualizations\\' + file_name)
    plt.show()
    ## example function call: side_by_side(all_data, 'Spain', 'Confirmed Cases', ['Linear Regression', 'Neural Network'], output, mlp, save_output=True)


def predicted_vs_real(full_df, output_df, country, target, logged=False, net=False):
    '''
    Plots one model's prediction vs real trend
    '''
    style.use('seaborn')
    date = output_df.index.min()
    pre = full_df[(full_df['Country'] == country) & (full_df['Date'] <= date)][[target, 'Date']]
    real = country + ' real'
    predict = country + ' prediction'
    post = output_df[[real, predict]]
    day = pre['Date'].max() + dt.timedelta(days=-1)
    row = pre[pre['Date'] == day][[target, 'Date']]
    val = row[target]
    if logged:
        row[real], row[predict] = (np.log(val), np.log(val))
    else:
        row[real], row[predict] = (val, val)
    row.set_index('Date', inplace=True)
    post = row.append(post[[real, predict]])
    if logged:
        post = np.exp(post)

    fig, ax = plt.subplots(figsize=(12, 8))
    title = 'Prediction: {} in {}'.format(target, country)
    if net:
        title = 'Neural Net ' + title
    else:
        title = 'Linear Regression ' + title
    plt.title(label=title, fontsize=15)
    ax.axvline(x=date, ls=':', c='gray', label = str(date))
    g = sns.lineplot(x=post.index, y=post[real], ax=ax, marker='X', color='darkorange')
    g = sns.lineplot(x=post.index, y=post[predict], ax=ax, marker='X', color='g')
    g = sns.lineplot(x=pre['Date'], y=pre[target], ax=ax, color='royalblue')
    plt.legend(('Prediction frontier\n {}'.format(date), 'Real', 'Predicted', 'Trend'), prop={'size': 12})
    plt.ylabel(target)
    plt.show()

    #example call: predicted_vs_real(all_data, mlp, 'Spain', 'Confirmed Cases', logged=False, net=True)