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
from selector.kp.kp_selector import select_kp_data_from_a_date_range
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

prepareParams = yaml.safe_load(open("params.yaml"))["prepare"]
evaluateParams = yaml.safe_load(open("params.yaml"))["evaluate"]

# obtain scaler from trainset
########################################
# load trainset and testset from data/preapred
########################################
trainsetPath = os.path.join(sys.argv[1])
testsetpath = os.path.join(sys.argv[2])
dfTrainset = pd.read_csv(trainsetPath)
dfTestset = pd.read_csv(testsetpath)


#######################################
# persist method
#######################################

# add elements from testset except the last
naivePersistDataset = list(dfTrainset['tec'][-12:])  #list(dfTestset['tec'][0:11])
# add last element from train set
#naivePersistDataset.insert(0, dfTrainset['tec'].iloc[-1])
# real values
realValues = dfTestset['tec'][0:12]

plt.title("TEC predicted 24 h. ahead on "+prepareParams['station']+" lat: "+str(evaluateParams['lat'])+", long: "+str(evaluateParams['long'])+" \n \
using naive methods (persistent)", fontsize=12, fontweight='bold')
plt.plot(realValues, label="Real TEC")
plt.plot(naivePersistDataset, label="TEC predicted")
plt.xlabel('UT', fontweight='bold')
plt.ylabel('TECu (el/m^2)', fontweight='bold')
plt.xticks(range(12),range(0,24,2))
#plt.margins(0)
plt.legend()
plt.grid()
plt.savefig('data/models/evaluation_model_naive_persistent.png')
plt.show()


print("\n Naive method persistent errors: ")
mse = mean_squared_error(realValues,naivePersistDataset)
rmse = sqrt(mse)
print("MSE one day forecasted = ", mse)
print("RMSE one day forecasted = ", rmse)



#######################################
# media and median methods
#######################################

# make dataset with real values
realValues = list(dfTestset['tec'][0:12])

# make dataset with predicted values
# amount of past days
amountOfDays = 5
amountOfDataPerDay = 12
dataSet = list(dfTrainset['tec'][-amountOfDataPerDay*amountOfDays:])
print("data set --> ", dataSet)

predictedValues = list()
for i in range(amountOfDataPerDay):
    aux = list()
    for j in range(amountOfDays):
        aux.append(dataSet[i+j*12])
    predictedValues.append(mean(aux))
print("predicted values --> ", predictedValues)
    
plt.title("TEC predicted 24 h. ahead on "+prepareParams['station']+" lat: "+str(evaluateParams['lat'])+", long: "+str(evaluateParams['long'])+" \n \
using naive methods (mean of past 12 values)", fontsize=12, fontweight='bold')
plt.plot(realValues, label="Real TEC")
plt.plot(predictedValues, label="TEC predicted")
plt.xlabel('UT', fontweight='bold')
plt.ylabel('TECu (el/m^2)', fontweight='bold')
plt.xticks(range(12),range(0,24,2))
#plt.margins(0)
plt.legend()
plt.grid()
plt.savefig('data/models/evaluation_model_naive_mean.png')
plt.show()

print("\n Naive method mean errors: ")
mse = mean_squared_error(realValues,predictedValues)
rmse = sqrt(mse)
print("MSE one day forecasted = ", mse)
print("RMSE one day forecasted = ", rmse)



'''
########################
# ts shifted
########################

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

plt.subplot(211)
plt.title("TEC predicted 24 h. ahead on "+prepareParams['station']+" [lat.: "+str(evaluateParams['lat'])+", long.: "+str(evaluateParams['long'])+"]", fontsize=12, fontweight='bold')
plt.plot(realValues[0:12], label= 'Real TEC')
plt.plot(predictions[13:25], label= 'TEC predicted')
plt.xlabel('UT')
plt.ylabel('TECU (el/m^2)')
plt.legend()
plt.grid(True)
plt.xticks(range(12),range(0,24,2))
#plt.margins(0)

plt.subplot(212)
plt.title("TEC predicted for year 2016 on "+prepareParams['station']+" [lat.: "+str(evaluateParams['lat'])+", long.: "+str(evaluateParams['long'])+"]", fontsize=12, fontweight='bold')
plt.plot(realValues[:-1], label="Real TEC")
plt.plot(predictions[13:], label="TEC predicted")
plt.xlabel('UT')
plt.ylabel('TECU (el/m^2)')
plt.legend()
plt.grid(True)
#plt.margins(0)

plt.savefig('data/models/evaluation_model.png')

plt.show()


print("\n errors for TS shifted: ")
mse = mean_squared_error(realValues[0:12],predictions[13:25])
rmse = sqrt(mse)
print("MSE one day forecasted = ", mse)
print("RMSE one day forecasted = ", rmse)

# calculate overall RMSE
mse = mean_squared_error(realValues[:-1],predictions[13:])
rmse = sqrt(mse)
print("MSE for all testset = ", mse)
print("RMSE for all testset = ", rmse)


print("\n")
'''

