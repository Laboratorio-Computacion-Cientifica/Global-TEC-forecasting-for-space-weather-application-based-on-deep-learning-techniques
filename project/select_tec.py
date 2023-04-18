#!/usr/bin/env python
"""Select data from IONEX files for specific stations.

"""
from config import tec_config as tc
from selector.tec import tec_selector as ts
from general_functions import date_functions
import numpy as np
import sys
import pandas as pd
np.set_printoptions(threshold=sys.maxsize)

localDestinyDirectoryForFilteredDataByStation = \
    '../data/tec/preprocessed_tec'

# years of interest
_years = [2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]

for _year in _years:
    # calculate amount of days of the year
    amountOfDaysOfTheYear = \
        date_functions.calculate_amount_of_days_of_an_year(_year)

    # dataframe for load all the data of the stations of interest
    dfForDataOfSatationsOfInterest = pd.DataFrame()

    for day in range(1, amountOfDaysOfTheYear+1):
        # get tec maps of one day from a local file
        _filename = \
            ts.ionex_local_path(_year, day, tc.centre,
                                tc.localDestinyDirectoryForIonexFiles +
                                '/tec'+str(_year)+'/')
        print(_filename)
        tecmap = ts.get_tecmaps(_filename, len(tc.stationsOfInterest))
        # get tec of the stations of interest
        _indexForStation = 1
        # dataframe for load current data
        _dfAux = pd.DataFrame()
        _date = _filename[-8:-4] + _filename[-3:-1]        
        for _element in tc.stationsOfInterest:
            _latitude = tc.stationsOfInterest[_element][0]
            _longitude = tc.stationsOfInterest[_element][1]
            # check if tecmap is an existing file, else, use nans
            if(tecmap):
                tecValues = [ts.get_tec(t, _latitude, _longitude)
                             for t in tecmap]
                tecValues = [round(_element, 2) for _element in tecValues]
            else:
                print("Missed file: {}".format(_filename))
                # 13 samples per day of each station
                tecValues = [np.nan]*13
            _station = "station"+str(_indexForStation)
            _dfAux['DATE'] = str(_date)
            _dfAux[_station] = tecValues
            _indexForStation += 1
        dfForDataOfSatationsOfInterest = dfForDataOfSatationsOfInterest.append(
            _dfAux, ignore_index=True)
    dfForDataOfSatationsOfInterest.to_csv(
        localDestinyDirectoryForFilteredDataByStation + '/' +
        str(_year)+'filtered_by_station.txt')
