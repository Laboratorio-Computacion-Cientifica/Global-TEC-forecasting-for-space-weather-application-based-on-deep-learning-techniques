#!/usr/bin/env python
"""Selector module for select data from Kp Index files.

"""

import pandas as pd
from numpy import array

def load_dataframe(path):
    return pd.read_csv(path)

def get_index_of_row(df, column, value):
    '''
        This function return the index of a row.
        ----------
        input
        -----
        df: dataframe involved (DataFrame).
        column: name of the column involved (string).
        value: value for the query (any).
        Output
        ------
        index: the index of a row (int64).
    '''    
    return df[df[column]==value].index

def select_kp_data_from_a_date_range(path, initDate, endDate):
    # load dataframe
    df = load_dataframe(path)
    # get index of rows with init date and end date
    index1 = get_index_of_row(df, 'YYYYMMDD', initDate)
    index2 = get_index_of_row(df, 'YYYYMMDD', endDate)
    # get a slice of the dataframe using indexes
    return df.iloc[index1[0]:index2[0]+1]

def transform_kp_dataframe_in_array(df):
    '''
        This function select the kp values per row.
        ----------
        input
        -----
        Output
        ------
    '''    
    listAux = list()
    for row in df.itertuples():
        # init in 3, because itertuples add an index column
        for i in range(3,15):
            listAux.append(round(row[i],2))
    return array(listAux)


'''

path = '/home/jorge/Desktop/Doctorado/work_with_italy/repo_ingv_tec_prediction/data/kp/preprocessed_kp/kp_index_filtered_taking_into_account_signs_interpolated.txt'
initDate = 20100101
endDate = 20161231

df = select_kp_data_from_a_date_range(path, initDate, endDate)
print(df.shape)
listAux = transform_kp_dataframe_in_array(df)
print(len(listAux))
'''
