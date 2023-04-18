#!/usr/bin/env python
"""Implementation of ...

"""


import pandas as pd
from selector.kp import kp_selector
from config import general_configs as gc

path = gc.dataFolder + '/data/kp/preprocessed_kp/kp_index_filtered_taking_into_account_signs_interpolated.txt'
initDate = 20050101
endDate = 20161231

df = kp_selector.select_data_from_a_date_range(path, initDate, endDate)
