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
import numpy as np

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

print(dfTestset.iloc[1])

########################################
# genetare scalers from trainset
########################################
dfTrainsetToNumpy = dfTrainset.to_numpy()
trainTecScaledData, scalerTec = scale_data(dfTrainsetToNumpy[:,1].reshape(-1,1),"normalization")
trainKpScaledData, scalerKp = scale_data(dfTrainsetToNumpy[:,2].reshape(-1,1),"normalization")

# transform trainset data
# convert to [rows, columns] structure
in_seq1 = trainTecScaledData.reshape((len(trainTecScaledData), 1))
in_seq2 = trainKpScaledData.reshape((len(trainKpScaledData), 1))
# horizontally stack columns
trainDataset = hstack((in_seq1, in_seq2))


########################################
# scaling testset
########################################
dfTestsetToNumpy = dfTestset.to_numpy()
# using the scalers of trainsets
testTecScaledData  = scalerTec.transform(dfTestsetToNumpy[:,1].reshape(-1,1))
testKpScaledData = scalerKp.transform(dfTestsetToNumpy[:,2].reshape(-1,1))

# transform trainset data
# convert to [rows, columns] structure
in_seq1 = testTecScaledData.reshape((len(testTecScaledData), 1))
in_seq2 = testKpScaledData.reshape((len(testKpScaledData), 1))
# horizontally stack columns
testDataset = hstack((in_seq1, in_seq2))


print('Testset shape: ', testDataset.shape)



########################################
# load model
########################################
model = load_model(sys.argv[3])


########################################
# take last n steps from trainset, predict and compare with first n stepas of testset
########################################
# unscaled data for plot
realValues = list()
# apend last train values
#realValues.append([nan] * 12)
# append test values
print("1->", dfTestsetToNumpy.shape)
print("2->", dfTestsetToNumpy.shape[0]-12)
for i in range(0,dfTestsetToNumpy.shape[0]-12,12):
    # take tec values, column 2, index 1
    if(i+12<dfTestsetToNumpy.shape[0]-12):
        realValues.append(dfTestsetToNumpy[i:i+12,1].tolist())
print("4->", len(realValues))
realValues = [item for sublist in realValues for item in sublist]

print("real values len: ",len(realValues))

# scaled data for predictions and plots
predictions = list()
# making predictions over last values of trainset
# note: the split with window size is taking out the last values of trainset
# i need to take the last 8 values of trainset and the first 4 values of testset
#X = trainDataset[-12:,:]
X = trainDataset[-12:,:]
X = X.reshape(1, X.shape[0], X.shape[1])
yhat = model.predict(X, verbose=0)
yhat = unscale_data(yhat[0].reshape(-1,1),scalerTec)
predictions.append(yhat.tolist())

# making predictions over testset
for i in range(0,dfTestsetToNumpy.shape[0]-12,12):
    # in numpy the selection include the last values.
    if(i+12<dfTestsetToNumpy.shape[0]-12):
        X = testDataset[i:i+12]
        X = X.reshape(1, X.shape[0], X.shape[1])
        yhat = model.predict(X, verbose=0)
        yhat = unscale_data(yhat[0].reshape(-1,1),scalerTec)
        predictions.append(yhat.tolist())
predictions = [item for sublist in predictions for item in sublist]

print("predictions len: ",len(predictions))

'''
plt.plot(realValues, label="Real TEC")
plt.plot(predictions, label="TEC predicted")
#plt.title("TEC predicted for years 2013 and 2014 on station 1 [lat.: -85, long.: -120]", fontsize=14, fontweight='bold')
plt.title("TEC predicted for year 2014 on station 1 [lat.: -85, long.: -120]", fontsize=14, fontweight='bold')
plt.legend()
plt.ylabel("[TECu]")
plt.margins(0)
plt.grid()
plt.show()
'''


########################
# ts shifted
########################

fig, axs = plt.subplots(1, 1, figsize=(6, 3), tight_layout=False)
#plt.figure(figsize=(5, 3), dpi=200, constrained_layout=True)
#plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.4f}'))
#plt.title(station, fontsize=20, fontweight='bold')
axs.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
axs.yaxis.get_offset_text().set_fontsize(20)
axs.tick_params(axis='both', labelsize=20)
#axs.set_xlim(0, 21)
#axs.set_xticks([0,2,4,6,8,10,12])
#axs.set_xticklabels([0,4,8,12,16,20,22])
axs.set_xticks([0,4,8,12,16,20,22])
axs.set_xticklabels([0,4,8,12,16,20,22])
domain = np.arange(0,23,2)
axs.plot(domain, realValues[0:12],linewidth = '5')
axs.plot(domain, predictions[13:25], linewidth = '5')
axs.margins(0)
axs.grid()
fig.savefig('data/models/evaluation_model.png')
plt.show()



'''
plt.title("TEC predicted 24 h. ahead on "+prepareParams['station']+" lat: "+str(evaluateParams['lat'])+", long: "+str(evaluateParams['long'])+" \n \
using LSTM", fontsize=12, fontweight='bold')
plt.plot(realValues[0:12], label= 'Real TEC')
plt.plot(predictions[13:25], label= 'TEC predicted')
plt.xlabel('UT', fontweight='bold')
plt.ylabel('TECU (el/m^2)', fontweight='bold')
plt.legend()
plt.grid(True)
plt.xticks(range(12),range(0,24,2))
plt.savefig('data/models/evaluation_model.png')
plt.margins(0)
plt.show()
'''


'''
# the next code is working ok!

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
'''

print("\n errors for LSTM TS shifted: ")
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

########################
# ts not shifted
########################

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

plt.subplot(211)
plt.title("TEC predicted for 2016/01/01 on "+prepareParams['station']+" [lat.: "+str(evaluateParams['lat'])+", long.: "+str(evaluateParams['long'])+"]", fontsize=12, fontweight='bold')
plt.plot(realValues[0:12], label= 'Real TEC')
plt.plot(predictions[0:12],label= 'TEC predicted')
plt.xlabel('UT')
plt.ylabel('TECU (el/m^2)')
plt.legend()
plt.grid(True)
plt.xticks(range(12),range(0,24,2))
#plt.margins(0)

plt.subplot(212)
plt.title("TEC predicted for year 2016 on "+prepareParams['station']+" [lat.: "+str(evaluateParams['lat'])+", long.: "+str(evaluateParams['long'])+"]", fontsize=12, fontweight='bold')
plt.plot(realValues[:-1], label="Real TEC")
plt.plot(predictions[:-13], label="TEC predicted")
plt.xlabel('UT')
plt.ylabel('TECU (el/m^2)')
plt.legend()
plt.grid(True)
#plt.margins(0)

plt.show()


print("\n errors for TS not shifted: ")
mse = mean_squared_error(realValues[0:12],predictions[0:12])
rmse = sqrt(mse)
print("MSE one day forecasted = ", mse)
print("RMSE one day forecasted = ", rmse)

# calculate overall RMSE
mse = mean_squared_error(realValues[:-1],predictions[:-13])
rmse = sqrt(mse)
print("MSE for all testset = ", mse)
print("RMSE for all testset = ", rmse)
'''
