# -*- coding: utf-8 -*-
"""
Created on Wed May  6 16:55:31 2020

@author: diego
"""

import pandas as pd
import numpy as np


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
                            header=3, usecols = ["Country Name", "2018"])
    hbeds = pd.read_excel(data_dir+'hospbeds.xls', sheet_name='Data',\
                          header=3)
    yrs = hbeds[hbeds.columns[4:]]
    yrs = yrs.ffill(axis=1)['2019']
    wb_data = pd.concat([hbeds['Country Name'], yrs, pop_old['2018']], axis=1)
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