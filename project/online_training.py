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


# options:
# 1- re train the model for the next 24 hs only
# 2- walking forward retraining the model


################################
# dev for the option
# 1- re train the model for the next 24 hs only

########################################
# load trainset and testset from data/preapred
########################################
trainsetPath = os.path.join(sys.argv[1])
testsetpath = os.path.join(sys.argv[2])
dfTrainset = pd.read_csv(trainsetPath)
dfTestset = pd.read_csv(testsetpath)


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


# scaling testset
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

print('Head ofTestset : ', testDataset[0:10])
print("testSet shape", testDataset.shape)


########################################
# load model
########################################
model = load_model(sys.argv[3])


# predict without re train

# take next 24 h of tec and kp
X_new = testDataset[0:12]
y_new = testDataset[12:24][:,0]
print("x shape: ", X_new.shape)
print("y new ", y_new.shape)

print("predict without re train \n")
X_new = X_new.reshape(1,12,2)
yhat = model.predict(X_new, verbose=0)
yhat = unscale_data(yhat[0].reshape(-1,1),scalerTec)
y_new = unscale_data(y_new.reshape(-1,1),scalerTec)
# calculate errors
mse = mean_squared_error(y_new,yhat)
rmse = sqrt(mse)
print("MSE one day forecasted = ", mse)
print("RMSE one day forecasted = ", rmse)
plt.plot(y_new)
plt.plot(yhat)
plt.show()

###################################################

# predict using re training
# take next 24 h of tec and kp
X_new = testDataset[0:12]
y_new = testDataset[12:24][:,0]
print("x shape: ", X_new.shape)
print("y new ", y_new.shape)

# convert x_new to [samples, timesteps, features]
X_new = X_new.reshape(1,12,2)
y_new = y_new.reshape(1,12)

# retrain the model using new data
model.fit(X_new, y_new, epochs=10, batch_size=32, verbose=2)

# evaluate new model, make predictions
yhat = model.predict(X_new, verbose=0)
yhat = unscale_data(yhat[0].reshape(-1,1),scalerTec)
y_new = unscale_data(y_new.reshape(-1,1),scalerTec)

# calculate errors
print("\n Errors for LSTM: ")
mse = mean_squared_error(y_new,yhat)
rmse = sqrt(mse)
print("MSE one day forecasted = ", mse)
print("RMSE one day forecasted = ", rmse)

fig, axs = plt.subplots(1, 1, figsize=(6, 3), tight_layout=False)
axs.set_title("Forecast 24 h. ahead updating model on new data only", fontsize=12, fontweight='bold')
axs.plot(y_new)
axs.plot(yhat)
axs.set_xticks([0,2,4,6,8,10,12])
axs.set_xticklabels([0,4,8,12,16,20,22],fontweight='bold', fontsize=12)
axs.set_xlabel("UT", fontweight='bold', fontsize=12)
axs.set_ylabel('TECU (el/m^2)', fontweight='bold', fontsize=12)

axs.margins(0)
plt.show()