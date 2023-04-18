#!/usr/bin/env python
"""Selector module for select data from IONEX files.

"""
import os, glob
import re
import numpy as np
import pandas as pd


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


def transform_tec_dataframe_in_array_and_select_station(df, station='station1'):
    listAux = list()
    for index, row in df.iterrows():
        listAux.append(round(row[station],2))
    return np.array(listAux)
