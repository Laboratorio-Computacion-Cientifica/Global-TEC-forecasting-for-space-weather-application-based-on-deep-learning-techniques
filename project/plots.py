#!/usr/bin/env python
"""Implementation of models.

"""


# multivariate lstm example
import os
import sys
from numpy import array
from numpy import hstack
import yaml
import config.general_configs as gc
from pre_processing_data.splitters.multivariate_splitters import sliding_window
from selector.kp.kp_selector import select_kp_data_from_a_date_range
from selector.kp.kp_selector import transform_kp_dataframe_in_array
from selector.tec.tec_selector import build_a_tec_df_from_several_files
from selector.tec.tec_selector import build_a_tec_df_from_date_range
from selector.tec.tec_selector import transform_tec_dataframe_in_array_and_select_station
from pre_processing_data.splitters.multivariate_splitters import sliding_window
from matplotlib import pyplot as plt


params = yaml.safe_load(open("params.yaml"))["prepare"]

# path of preprocessed Kp data
kpFile = os.path.join(sys.argv[1])
# path of preprocessed TEC data
tecFolder = os.path.join(sys.argv[2])

#kpFile = gc.dataFolder + '/data/kp/preprocessed_kp/kp_index_filtered_taking_into_account_signs_interpolated.txt'
#tecFolder = gc.dataFolder + '/data/tec/preprocessed_tec/'


########################################
# load data
########################################
# define input sequences for Kp index and TEC

# input sequence for Kp index
# define init date and end date for slice the dataframe
initDate = params["dateInit"]
endDate = params["dateEnd"]
# slice dataframe
dfKp = select_kp_data_from_a_date_range(kpFile, initDate, endDate)
print("\nDimension of Kp Index dataframe: {}".format(dfKp.shape))
# transform dataframe to array
in_seq1 = transform_kp_dataframe_in_array(dfKp)
print("Kp Index data transformed to array:  {}".format(in_seq1))

# input sequence for TEC
# build one file with all the data
yearInit = int(str(initDate)[0:4])
yearEnd = int(str(endDate)[0:4])

dfTec = build_a_tec_df_from_date_range(tecFolder, yearInit, yearEnd)
print("\nDimension of TEC dataframe: {}".format(dfTec.shape))
#print(dfTec.head())
in_seq2 = transform_tec_dataframe_in_array_and_select_station(dfTec, 'station1')
print("TEC data transformed to array:  {}".format(in_seq2))

########################################
# transform data
########################################
# output sequence. In this case, is equal to TEC st
out_seq = in_seq1
print("\nOutput sequence (TEC): {}".format(out_seq))

# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))

# source multiple plots: https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplot.html

fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
fig.suptitle('Plots in range ['+str(yearInit)+'/01/01-31/12/'+str(yearEnd)+']: 1- Kp Index - 2- TEC - 3- Kp Index + TEC', fontsize=12, fontweight='bold')

ax1.plot(in_seq1, label= 'Kp index')
ax1.set_xlabel('')
ax1.set_ylabel('Kp Index')
ax1.legend()
ax1.grid(True)
ax1.margins(0)

ax2.plot(in_seq2, label= 'TEC')
ax2.set_xlabel('')
ax2.set_ylabel('TECU (el/m^2)')
ax2.legend()
ax2.grid(True)
ax2.margins(0)

ax3.plot(in_seq1, label= 'Kp Index')
ax3.plot(in_seq2, label= 'TEC')
ax3.set_xlabel('')
ax3.set_ylabel('')
ax3.legend()
ax3.grid(True)
ax3.margins(0)

plt.show()
