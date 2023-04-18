#!/usr/bin/env python
"""Download TEC files from a URL source.

"""
from config import tec_config as tc
from downloader.tec import tec_downloader as td
from general_functions import date_functions

# loop over the days of the year of interest
_years = [2017] #[2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]

for _year in _years:
    _amountOfDaysOfTheYear = \
        date_functions.calculate_amount_of_days_of_an_year(_year)
    for day in range(1, _amountOfDaysOfTheYear+1):
        # download the file of each day
        td.download_ionex(_year, day, tc.ftpServer, tc.ftpDirectory,
                          tc.ftpCredencialEmail, tc.centre,
                          tc.localDestinyDirectoryForIonexFiles+str(_year)+'/')
