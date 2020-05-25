# -*- coding: utf-8 -*-
"""
Created on Wed May  6 16:55:31 2020

@author: diego
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import datetime


def metrics(target_predict,test_targets, train_features, train_targets,
            model, output=True):

    bias = mean_squared_error(model.predict(train_features),train_targets)
    mse = mean_squared_error(target_predict,test_targets)
    rss = np.sum((target_predict - test_targets) ** 2)
    variance = model.score(train_features, train_targets)

    if output:
        print("Bias: %.2f" % bias)
        print("Mean squared error: %.2f" % mse)
        print("RSS: %.2f" % rss)
        print('Variance score: %.2f\n' % variance)

    return(bias,mse,rss,variance)


def get_most_relevant_features(df, model, number_of_features):
    '''


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
                               model.intercept_, 'abs': abs(model.intercept_)}
                               , ignore_index=True)
    for i in range(len(df.columns)):
        features = features.append({'Feature': df.columns[i],
                                    'Coefficient': model.coef_[i],
                                    'abs': abs(model.coef_[i])},
                                    ignore_index=True)

    features = features.sort_values(by=['abs'], ascending=False)
    features = features.drop(['abs'], axis=1)
    return features[:number_of_features]


def read_and_process_data(filepath):
    '''


    Returns
    -------
    None.

    '''
    df_train, df_test = read_split_and_scale(filepath)
    df_train['Country'].astype('category')
    df_test['Country'].astype('category')
    df_train = df_train.drop(['Date'], axis=1)
    df_test = df_test.drop(['Date'], axis=1)
    df_train = pd.get_dummies(df_train)
    df_test = pd.get_dummies(df_test)
    df_train = remove_countries_not_in_test_set(df_train, df_test)
    return df_train, df_test


def read_split_and_scale(filepath):
    '''


    Returns
    -------
    None.

    '''
    df = pd.read_pickle(filepath)
    split_date = datetime.date(2020, 5, 1)
    df_training, df_test = split_train_test_on_date(df, split_date)
    scalable_train, non_scalable_train = split_scalable_columns(df_training)
    scalable_test, non_scalable_test = split_scalable_columns(df_test)
    scaled_train, scaled_test = scale_df(scalable_train, scalable_test)
    '''
    df_train = scaled_train.join(non_scalable_train)
    df_test = scaled_test.join(non_scalable_test)
    '''
    df_train = pd.concat([scaled_train, non_scalable_train], axis=1)
    df_test = pd.concat([scaled_test, non_scalable_test], axis=1)
    return df_train, df_test


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
    non_scalable_vars = df[['Country', 'Date', 'Day Count',
                            'Days Elapsed Since First Case']]
    lst = []
    for column in df.columns:
        if column not in non_scalable_vars.columns:
            lst.append(column)
    scalable_vars = df[lst]
    return scalable_vars, non_scalable_vars


def split_train_test_on_date(df, split_date):
    '''


    Parameters
    ----------
    split_date : Datetime date.

    Returns
    -------
    train_df, test_df.

    '''
    split_date = datetime.date(2020, 5, 1)
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

    '''
    scaler = preprocessing.StandardScaler().fit(scalable_train)
    scaled_train = pd.DataFrame(scaler.transform(scalable_train),
                                columns=scalable_train.columns.values)
    scaled_test = pd.DataFrame(scaler.transform(scalable_test),
                               columns=scalable_test.columns)
    return scaled_train, scaled_test


def sanity_check(train_df, test_df):

    # Sort features alphabetically
    train_df = train_df.reindex(sorted(train_df.columns), axis=1)
    test_df = test_df.reindex(sorted(test_df.columns), axis=1)

    # Check that they have the same features
    if (train_df.columns == test_df.columns).all():
        print("Success: Features match")
    '''
    # Check that no NAs remain
    if  not train_df.isna().sum().astype(bool).any() and \
        not test_df.isna().sum().astype(bool).any():
        print("Success: No NAs remain")
    '''


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
    x = df.drop([target], axis=1)
    return x, y


def train_and_evaluate(X_train, X_test, y_train, y_test, models, grid):
    '''


    Parameters
    ----------
    X_train : TYPE
        DESCRIPTION.
    X_test : TYPE
        DESCRIPTION.
    y_train : TYPE
        DESCRIPTION.
    y_test : TYPE
        DESCRIPTION.
    models : TYPE
        DESCRIPTION.
    grid : TYPE
        DESCRIPTION.

    Returns
    -------
    results_df : TYPE
        DESCRIPTION.

    '''
    results_df = pd.DataFrame({"Model": [], "Params": [], "R2 Score": []})
    counter = 0
    # Loop over models
    for model_key in models.keys():

        # Loop over parameters
        for params in grid[model_key]:
            print("Training model:", model_key, "|", params)

            # Create model
            model = models[model_key]
            model.set_params(**params)
            # Fit model on training set
            model = model.fit(X_train, y_train)

            # Predict on testing set
            y_predicted = model.predict(X_test)

            # Evaluate predictions
            accuracy_s, f1 = r2_score(y_test, y_predicted)

            # Store results in your results data frame
            results_df.loc[counter] = \
                [model_key, params, accuracy_s, f1]
            counter += 1

    return results_df


#%%
###
###
### READING AND MERGING FUNCTIONS
###
###

countries_names = {'Czech republic': 'Czech Republic', 'Czechia':
    'Czech Republic', 'Myanmar': 'Burma', 'West Bank and Gaza': 'Palestine',
    'Brunei Darussalam': 'Brunei', 'Korea Republic of' : 'South Korea',
    'Korea, Rep.' : 'South Korea', 'Korea, South' : 'South Korea',
    'Cote d\'Ivoire': 'Côte d\'Ivoire', 'North Macedonia Republic Of':
    'North Macedonia', 'Congo': 'Congo (Brazzaville)', 'Congo DR':
    'Congo (Kinshasa)', 'Congo, Dem. Rep.':'Congo (Kinshasa)', "Congo, Rep.":
    'Congo (Brazzaville)',  'Russian Federation': 'Russia', 'Egypt, Arab Rep.':
    'Egypt', 'Micronesia, Fed. Sts.':'Micronesia','Moldova Republic of':
    'Moldova', 'Moldova Republic Of': 'Moldova', 'Lao PDR': 'Laos', 'Viet Nam':
    'Vietnam', 'US': 'United States of America', "Bahamas, The": "Bahamas",
    "Gambia, The" : "Gambia", 'Iran, Islamic Rep.': "Iran", "Kyrgyzstan":
    "Kyrgyz Republic", 'St. Lucia' : "Saint Lucia", "Slovakia" :
    "Slovak Republic", 'Syrian Arab Republic' : "Syria", 'United States':
    'United States of America', 'St. Vincent and the Grenadines':
    'Saint Vincent and the Grenadines', 'Venezuela, RB': 'Venezuela',
    'Yemen, Rep.' : 'Yemen'}

data_dir = "../data/"
jhu_data_url = "https://raw.githubusercontent.com/datasets/covid-19/master/data\
/time-series-19-covid-combined.csv"
jhu_data_offline = data_dir + "time-series-19-covid-combined.csv"
acaps_filepath = data_dir + 'acaps_data.xlsx'
OXFORD_URL = "https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv"


def read_jhu_data(url):
    '''
    Reads John Hopkins cross country data, converts dates to datetime and
    returns a pandas dataframe
    '''
    try:
        jhu_df = pd.read_csv(jhu_data_url)
    except:
        jhu_df = pd.read_csv(jhu_data_offline)
    jhu_df['Date'] = pd.to_datetime(jhu_df['Date'], infer_datetime_format=True)
    jhu_df['Date'] = jhu_df['Date'].dt.date
    jhu_df.rename(columns={'Country/Region': 'Country'}, inplace=True)
    return jhu_df

def get_first_case(jhu_df):
    '''
    From John Hopkins data, gets the date of first case for each country and
    returns as a two column dataframe
    '''
    sub = jhu_df[jhu_df['Confirmed'] > 0.0]
    first_case = sub.groupby('Country')['Date'].min().reset_index().\
    sort_values(by=['Country'])
    return first_case


def read_acaps_data(file_path):
    '''
    Reads acaps data, changes dates to datetime, implements some data cleaning
    '''
    acaps_df = pd.read_excel(file_path, index_col=0,\
                             converters={'Date': str})

    acaps_df['DATE_IMPLEMENTED'] = pd.to_datetime(acaps_df['DATE_IMPLEMENTED'],\
            infer_datetime_format=True)
    acaps_df['DATE_IMPLEMENTED'] = acaps_df['DATE_IMPLEMENTED'].dt.date
    acaps_df.rename(columns={'COUNTRY': 'Country'}, inplace=True)

    #inaccurate date of decl of emergency in US, removing
    acaps_df = acaps_df.drop([acaps_df.index[4292] , acaps_df.index[4298]])
    #this should be done differently

    return acaps_df

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

def merge_oxford_jhu(oxford_df, jhu_df):
    return pd.merge(oxford_df, jhu_df, on=['Country', 'Date'], how='left')

def merge_dfs_jhu_acap(df1, df2):
    '''
    Merges John Hopkins df with acaps, obtaining dummies for some policies first
    '''
    df1['Country'] = df1['Country'].replace(countries_names)
    first_case = get_first_case(df1)
    #next command gets the number of policies from acaps dataset
    policies = df2.groupby('Country').size().reset_index(name='n_policies')
    df_merged = pd.merge(policies, first_case, on='Country', how='outer')
    subset_curfews = df2[df2.MEASURE == "Curfews"]
    curfew = subset_curfews.groupby('Country').size().reset_index(name='count')
    df_merged['Curfew'] = np.where(df_merged['Country'].isin(curfew['Country'])\
             , 1, 0)

    emergency_countries = df2[df2.MEASURE == "State of emergency declared"]
    emergency_date = emergency_countries.groupby('Country')['DATE_IMPLEMENTED']\
    .min().reset_index()
    df_merged = pd.merge(df_merged, emergency_date, on='Country', how='left')
    df_merged.rename(columns={"DATE_IMPLEMENTED": "Emergency Date", 'Date':\
                              'First Case'}, inplace=True)
    return df_merged

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
#%%

def main():
    df1 = read_jhu_data(jhu_data_url)
    df2 = read_acaps_data(acaps_filepath)
    merged_df = merge_dfs_jhu_acap(df1, df2)
    df_wb = read_wb_data(data_dir)
    merged_df = pd.merge(merged_df, df_wb, how='inner', on='Country')
    return merged_df


## Country Names are not consistent  in any dataset
## World Bank Provides country codes so using it to make consistent

renaming_jhu_names = {'Bahamas': 'The Bahamas',
                        'Burma': 'Myanmar',
                        'Congo (Brazzaville)' : 'Congo',
                        'Congo (Kinshasa)':'Dem. Rep. Congo',
                         'Gambia': 'The Gambia',
                         'Korea, South':'Korea',
                         'North Macedonia':'Macedonia',
                         'Syria':'Syrian Arab Republic',
                         'US':'United States'}

def jhu_with_country_code():
    '''
    This function will take the JHU df and will add country codes as per WB
    '''

    df = read_jhu_data(jhu_data_url)
    df['Country'] = df['Country'].replace(renaming_jhu_names)
    wb_codes = pd.read_csv("../data/WB_Country_Codes.csv")[['Country Code', 'Short Name']]
    wb_codes.rename(columns = {'Short Name':'Country'}, inplace = True)
    result = pd.merge(df, wb_codes, on='Country', how='inner')
    # only a few small countries do not match with World Bank list, it's best to drop them
    # at this stage
    return result

def create_clean_df():
    df1 = jhu_with_country_code()
    #making df1 at country-date level
    df1 = df1.groupby(['Country','Country Code', 'Date']).agg({'Recovered':'sum',
                                                            'Confirmed': 'sum',
                                                            'Deaths':'sum'}).reset_index()
    df2 = read_acaps_data(acaps_filepath)
    df2.rename(columns = {'ISO':'Country Code','DATE_IMPLEMENTED': 'Date'}, inplace=True)
    ## Makidn acaps data also at a day level
    ## <<FIX MEEEEEEEEEEEWEEEEEEEEE>>
    # this needs a bit more thinking
    df2 = df2.drop_duplicates(['Country Code', 'Date'])

    m_df = pd.merge(df1, df2, how='left', left_on=['Country Code','Date'], right_on = ['Country Code', 'Date'])
    df_wb = read_wb_data(data_dir)
    m_df = pd.merge(m_df, df_wb, on = ['Country Code'])

    ## some cleaning on the m_df
    m_df.drop(columns =['Country_x', 'Country_y'], inplace=True)
    m_df['acaps_measure'] = pd.notnull(m_df['CATEGORY']).astype(int)

    #adding dummies for acaps 'CATEGORY' var
    m_df = m_df.join(pd.get_dummies(m_df['CATEGORY'], prefix= "acaps_cat_"))

    ## making days since first case
    first_case = m_df[m_df['Confirmed'] > 0.0].groupby('Country').agg({'Confirmed': 'cumcount'})
    first_case.rename(columns = {'Confirmed': 'days_since_first_case'}, inplace=True)
    m_df = m_df.join(first_case)
    return m_df