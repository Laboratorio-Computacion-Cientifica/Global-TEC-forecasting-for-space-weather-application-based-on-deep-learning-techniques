import io
import os
import sys
import yaml
from numpy import linspace
from numpy import nan
from numpy import hstack
from numpy import savetxt
from numpy import array
from numpy import split
from math import sqrt
from numpy import mean
from numpy import median
import pandas as pd 
from selector.tec.tec_selector import select_data_of_a_tec_dataframe_based_on_date_range_and_station
from selector.tec.tec_selector import build_a_tec_df_from_date_range
from selector.tec.tec_selector import build_a_tec_df_from_several_files
from selector.kp.kp_selector import transform_kp_dataframe_in_array
from selector.tec.tec_selector import transform_tec_dataframe_in_array_and_select_station
from pre_processing_data.splitters.split_sequence import train_test_split
from pre_processing_data.scalers.scaler import scale_data
from pre_processing_data.scalers.scaler import unscale_data
from pre_processing_data.splitters.multivariate_splitters import sliding_window
from keras.models import load_model
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

########################################
# load data
########################################
# define input sequences for Kp index and TEC

params = yaml.safe_load(open("params.yaml"))["evaluate_specific_date_range"]
params2 = yaml.safe_load(open("params.yaml"))["prepare"]
stationFeatures = yaml.safe_load(open("params.yaml"))["evaluate"]

# 2- select tec data based on date range

# input sequence for TEC
# path of preprocessed TEC data
tecFolder = sys.argv[1]
initDate = params["dateInit"]
endDate = params["dateEnd"]
station = params2["station"]

# build one file with all the data
# take all tec data
#dfTec = build_a_tec_df_from_several_files(tecFolder)
# take tec in a date range
yearInit = int(str(initDate)[0:4])
yearEnd = int(str(endDate)[0:4])
dfTec = build_a_tec_df_from_date_range(tecFolder, yearInit, yearEnd)
print("\nDimension of TEC dataframe: {}".format(dfTec.shape))

# select data from date range
listTec = select_data_of_a_tec_dataframe_based_on_date_range_and_station(dfTec, initDate, endDate, station)
# convert to array
in_seq2 = array(listTec)

print("TEC data transformed to array dimension:  {}".format(in_seq2.shape))
print("TEC data transformed to array:  {}\n".format(in_seq2))
print("-------------------------------------\n")

########################################
# persistent method
########################################


predictedValues = in_seq2[:-12]

fig, axs = plt.subplots(1, 1, figsize=(6, 3), tight_layout=False)
axs.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
axs.yaxis.get_offset_text().set_fontsize(20)
axs.tick_params(axis='both', labelsize=20)
domain = range(len(in_seq2))
axs.set_xticks([0,4,8,12,16,20,22])
axs.set_xticklabels([0,4,8,12,16,20,22])
import numpy as np
domain = np.arange(0,23,2)

axs.plot(domain, in_seq2[12:],linewidth = '5')
axs.plot(domain, predictedValues, linewidth = '5')
axs.margins(0)
axs.grid()
fig.savefig('data/models/evaluation_model_specific_range_'+str(initDate)+'_'+str(endDate)+'naive_persistent'+'.png')
plt.show()

# errors
print("\n errors for naive persistent in selected range")
mse = mean_squared_error(in_seq2[12:],predictedValues)
rmse = sqrt(mse)
print("MSE one day forecasted  = ", mse)
print("RMSE one day forecasted  = ", rmse)


########################################
# avg, median methid
########################################
dfTec = build_a_tec_df_from_date_range(tecFolder, yearInit, yearEnd)
print("\nDimension of TEC dataframe: {}".format(dfTec.shape))

# select data from date range
listTec = select_data_of_a_tec_dataframe_based_on_date_range_and_station(dfTec, params["dateInitNaiveAvg"], endDate, station)
# convert to array
in_seq2_avg = array(listTec)

print("TEC data transformed to array dimension:  {}".format(in_seq2_avg.shape))
print("TEC data transformed to array:  {}\n".format(in_seq2_avg))
print("-------------------------------------\n")

# calculate avg
def avg_point_to_point(amountOfDataPerDay, amountOfDays, dataset):
    predictedValues = list()
    for i in range(amountOfDataPerDay):
        aux = list()
        for j in range(amountOfDays):
            aux.append(dataset[i+j*amountOfDataPerDay])
        predictedValues.append(mean(aux))
    return predictedValues

amountOfDataPerDay = 12
amountOfDays = 27
naiveDatasetAvg27Days = list()
for i in range(2):
    auxList = avg_point_to_point(amountOfDataPerDay, amountOfDays, in_seq2_avg[i*amountOfDataPerDay:])
    naiveDatasetAvg27Days.append(auxList)


# transform list of list to list
predictedValues = list()
for l in naiveDatasetAvg27Days:
    for e in l:
        predictedValues.append(e)


fig, axs = plt.subplots(1, 1, figsize=(6, 3), tight_layout=False)
axs.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
axs.yaxis.get_offset_text().set_fontsize(20)
axs.tick_params(axis='both', labelsize=20)
axs.plot(in_seq2[12:],linewidth = '5')
axs.plot(predictedValues[12:], linewidth = '5')
axs.margins(0)
axs.grid()
fig.savefig('data/models/evaluation_model_specific_range_'+str(initDate)+'_'+str(endDate)+'naive_avg'+'.png')
plt.show()

# errors
print("\n errors for naive avg 27 days in selected range")
mse = mean_squared_error(in_seq2[12:],predictedValues[12:])
rmse = sqrt(mse)
print("MSE one day forecasted  = ", mse)
print("RMSE one day forecasted  = ", rmse)
