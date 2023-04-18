#!/usr/bin/env python
"""Selector module for select data from IONEX files.

"""
import os, glob
import re
import numpy as np
import pandas as pd
from datetime import datetime


def parse_map(tecmap, exponent=-1):
    tecmap = re.split('.*END OF TEC MAP', tecmap)[0]
    return np.stack([np.fromstring(line, sep=' ')
                     for line in re.split('.*LAT/LON1/LON2/DLON/H\\n',
                                          tecmap)[1:]])*10**exponent


def get_tecmaps(filename, amountOfStations=18):
    # try for catch missed file
    try:
        with open(filename) as f:
            ionex = f.read()
            try:
                tecMaps = [parse_map(t)
                           for t in ionex.split('START OF TEC MAP')[1:]]
                return tecMaps
            except IOError as e:
                print("incomplete file", e)
                return False
    except FileNotFoundError as e:
        print("File not found", e)
        return False


def ionex_filename(year, day, centre, zipped=True):
    return '{}{:03d}0.{:02d}i{}'.format(centre, day, year % 100, '.Z'
                                        if zipped else '')


def ionex_local_path(year, day, centre='esa', directory='/tmp',
                     zipped=False):
    return directory + ionex_filename(year, day, centre, zipped)


def get_tec(tecmap, lat, lon):
    i = round((87.5 - lat)*(tecmap.shape[0]-1)/(2*87.5))
    j = round((180 + lon)*(tecmap.shape[1]-1)/360)
    return tecmap[i, j]


def build_a_tec_df_from_several_files(folder):
    # loop folder
    fileList = os.listdir(folder)
    fileList.sort()
    # dataframe    
    df = pd.DataFrame()    
    # build dataframe
    for _file in fileList:
        dfAux = pd.read_csv(folder + '/' + _file, dtype={'DATE':str})
        df = df.append(dfAux, ignore_index = True)
    return df


def build_a_tec_df_from_date_range(folder, yearInit, yearEnd):
    # loop folder
    fileList = os.listdir(folder)
    fileList.sort()
    fileListFiltered = list()
    # select files of the date range
    for _year in range(yearInit,yearEnd+1):
        for _file in fileList:
            # compare first 4 character of the name file being year
            if(int(_file[0:4])==_year):
                fileListFiltered.append(_file)
    # empty dataframe
    df = pd.DataFrame()    
    # build dataframe
    for _file in fileListFiltered:
        dfAux = pd.read_csv(folder + '/' + _file, dtype={'DATE':str})
        df = df.append(dfAux, ignore_index = True)
    return df


def select_data_of_a_tec_dataframe_based_on_date_range_and_station(dfTec, initDate, endDate, station):
    '''
        This function implement the selection of a range of data based on
        a date range an station.
        ----------
        input
        -----
        dfTec: dataframe
        yearInit:
        yearEnd:
        ddInit:
        ddEnd:
        mmInit:
        mmEnd:
        station:

        Output
        ------
        scaledData: data scaled
        scaler: scaler fited for data
    '''
    # example of tec date format: 363017
    yyInit = int(str(initDate)[2:4])
    yyEnd = int(str(endDate)[2:4])
    mmInit = int(str(initDate)[4:6])
    mmEnd = int(str(endDate)[4:6])
    ddInit = int(str(initDate)[6:])
    ddEnd = int(str(endDate)[6:])
    # build datetime object
    date1 = datetime(yyInit, mmInit, ddInit)
    date2 = datetime(yyEnd, mmEnd, ddEnd)    
    # doy init and doy end
    doyInit = date1.timetuple().tm_yday
    doyEnd = date2.timetuple().tm_yday
    # year in format yy
    year = str(yyInit)
    # empty dataframe
    dfAux = pd.Series()    
    # loop doy range
    for i in range(doyInit, doyEnd+1):
        # example of tec date format: 363017
        tecDate = str(i)+'0'+year
        # tec data has 13 examples per day. Take only 12 (resolution of 2 h.)
        dfAux = dfAux.append(dfTec.query('DATE == @tecDate').iloc[0:12][station], ignore_index=True)
    return dfAux


def transform_tec_dataframe_in_array_and_select_station(df, station='station1'):
    '''
        This function return an array with tec values selected from an station.
        ----------
        input
        -----
        Output
        ------
    '''    
    listAux = list()
    # note: take only 12 values per day.
    # originally, tec has 13 values per day in the source files.
    i = 1
    for index, row in df.iterrows():
        if(i<13):
            # tec values are stored in columns by station
            listAux.append(round(row[station],2))
            i+=1
        else:
            i=1
    return np.array(listAux)
