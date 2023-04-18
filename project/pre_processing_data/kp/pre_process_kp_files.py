#!/usr/bin/env python
"""This script will to pre process the kp files

"""

import re
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor


def filter_values_of_kp_from_a_row_without_taking_into_account_sign(chain):
    '''
        This function filter values of a row of Kp values, removing the sign.
        ----------
        input
        -----
        chain: chain of values of Kp index (string -format: v1,v2,v3,...,vn).
        ------
        output: list with values filtered without sign.
    '''
    return [int(x) for x in re.findall(r"[\w']+", chain)]


def filter_values_of_kp_from_a_row_taking_into_account_sign(chain):
    '''
        This function filter values of a row of Kp values, taking into account
        the sign.
        ----------
        input
        -----
        chain: chain of values of Kp index (string -format: v1,v2,v3,...,vn).
        ------
        output: list with values filtered taking into account the sign (list).
    '''
    return re.findall(r"[0-9][+-]?", chain)


def apply_scale_expressed_in_thirds_of_a_unit(aList):
    '''
        This function apply the scale expressed in thirds of a unit to a
        Kp value.
        ----------
        input
        -----
        aList: list of values of Kp with sign
                  (list -format: v1+,v2,v3-,...,vn).
        ------
        output: list with values scaled (list).
    '''
    listAux = list()
    for element in aList:
        newValue = int(element[0])
        if(len(element) > 1):
            if(element[1] == '+'):
                newValue += 1/3
            if(element[1] == '-'):
                newValue += -1+2/3
        listAux.append(round(newValue, 2))
    return listAux


def interpolate(aList):
    '''
        This function interpolate values of Kp index by 2 hours.
        ----------
        input
        -----
        aList: list of values of Kp (list -size 8).
        ------
        output: list with values interpolated (list -size 12).
    '''
    # original points of kp, based in 3 hours
    # the last value 23, is for the interpolation of the hour 22
    X = [[0], [3], [6], [9], [12], [15], [18], [21], [24]]
    y = aList
    # config and fit interoplation
    neigh = KNeighborsRegressor(n_neighbors=2)
    neigh.fit(X, y)
    # new points for interpolate by 2 hours
    newX = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
    i = 0
    listAux = list()
    for element in newX:
        if([element] not in X):
            listAux.append(round(neigh.predict([[element]])[0], 2))
            i += 1
        else:
            listAux.append(y[i])
    return listAux


def read_kp_file_filter_and_save_new_file(path, destiny,
                                          takingAccountSign=True):
    '''
        This function read a Kp index file from a path, filter the data
        (with or without signs), and save the result in a destiny.
        ----------
        input
        -----
        path: path of the origin of the file (string).
        destiny: path of the destiny of the result (string).
        takingAccountSign aList: bool value (default: True).
        ------
        output: -
    '''
    # read file
    _file = open(path)
    # jump header
    _file.readline()
    # aux lists for save data filtered
    listAux0 = list()
    listAux2 = list()
    # loop file
    for row in _file:
        listAux1 = list()
        # conditional about sign
        if(takingAccountSign):
            listAux0 = filter_values_of_kp_from_a_row_taking_into_account_sign(row[9:25])
            listAux1 = apply_scale_expressed_in_thirds_of_a_unit(listAux0)
        else:
            listAux1 = filter_values_of_kp_from_a_row_without_taking_into_account_sign(row[9:25])
        # add date
        listAux1.insert(0, int(row[0:8]))
        # save date and data
        listAux2.append(listAux1)
    # convert list to dataframe
    df = pd.DataFrame(listAux2, columns=['YYYYMMDD', 'v0', 'v3', 'v6', 'v9', 'v12', 'v15', 'v18', 'v21'])
    # save dataframe
    df.to_csv(destiny)


def read_kp_file_interpolate_and_save_new_file(path, destiny):
    '''
        This function interpolate all the data of a Kp index file.
        ----------
        input
        -----
        path: path of the origin of the file (string).
        destiny: path of the destiny of the result (string).
        ------
        output: -
    '''
    df = pd.read_csv(path)
    listAux = list()
    for index, row in df.iterrows():
        aListInterpolated = list()
        # list with values for interpolate
        if(index < df.shape[1]):
            aList = [row['v0'], row['v3'], row['v6'], row['v9'], row['v12'], row['v15'], row['v18'], row['v21'], df.iloc[index+1]['v0']]
        else:
            aList = [row['v0'], row['v3'], row['v6'], row['v9'], row['v12'], row['v15'], row['v18'], row['v21'], row['v21']]
        # interpolate values of the list
        aListInterpolated = interpolate(aList)
        aListInterpolated.insert(0, int(df.iloc[index]['YYYYMMDD']))
        listAux.append(aListInterpolated)
    # convert list to dataframe
    df2 = pd.DataFrame(listAux, columns=['YYYYMMDD', 'v0', 'v2', 'v4', 'v6', 'v8', 'v10', 'v12', 'v14', 'v16', 'v18', 'v20', 'v22', 'v24'])
    # save dataframe
    df2.to_csv(destiny)


########################################################################
########################################################################

# filtering kp index
'''
path = '../../../data/kp/raw_data/kp_index.txt'
destiny = '../../../data/kp/preprocessed_kp/kp_index_filtered_taking_into_account_signs.txt'
read_kp_file_filter_and_save_new_file(path, destiny, True)
'''

# interpolating kp
path2 = '../../../data/kp/preprocessed_kp/kp_index_filtered_taking_into_account_signs.txt'
destiny2 = '../../../data/kp/preprocessed_kp/kp_index_filtered_taking_into_account_signs_interpolated.txt'
read_kp_file_interpolate_and_save_new_file(path2, destiny2)
