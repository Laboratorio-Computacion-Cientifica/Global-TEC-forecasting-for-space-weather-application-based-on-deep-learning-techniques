#!/usr/bin/env python
"""Functions about dates.

"""
import datetime


def calculate_amount_of_days_of_an_year(year):
    # calculate amount of days of the year
    d1 = datetime.datetime(year, 1, 1)
    d2 = datetime.datetime(year, 12, 31)
    return (d2-d1).days+1
