#!/usr/bin/env python
"""Implementation of models.

"""


# multivariate lstm example
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import config.general_configs as gc
from pre_processing_data.splitters.multivariate_splitters import sliding_window
from selector.kp.kp_selector import select_kp_data_from_a_date_range
from selector.kp.kp_selector import transform_kp_dataframe_in_array
from selector.tec.tec_selector import build_a_tec_df_from_several_files
from selector.tec.tec_selector import transform_tec_dataframe_in_array_and_select_station
from pre_processing_data.splitters.multivariate_splitters import sliding_window


########################################
# load data
########################################
# define input sequences for Kp index and TEC

# input sequence for Kp index
# path of preprocessed Kp data
kpFile = gc.dataFolder + '/data/kp/preprocessed_kp/kp_index_filtered_taking_into_account_signs_interpolated.txt'
# define init date and end date for slice the dataframe
initDate = 20050101
endDate = 20161231
# slice dataframe
dfKp = select_kp_data_from_a_date_range(kpFile, initDate, endDate)
print("\nDimension of Kp Index dataframe: {}".format(dfKp.shape))
# transform dataframe to array
in_seq1 = transform_kp_dataframe_in_array(dfKp)
print("Kp Index data transformed to array:  {}".format(in_seq1))

# input sequence for TEC
# path of preprocessed TEC data
tecFolder = gc.dataFolder + '/data/tec/preprocessed_tec/'
# build one file with all the data
dfTec = build_a_tec_df_from_several_files(tecFolder)
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

########################################
# split data
########################################
# choose a number of time steps
nStepsIn, nStepsOut = 13, 13
# convert into input/output
X, y = sliding_window(dataset, nStepsIn, nStepsOut)
# the dataset knows the number of features, e.g. 2
n_features = X.shape[2]

print(X.shape)
print(y.shape)

########################################
# scaling and split data data in train, test and validation subsets
########################################

# ¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡




########################################
# keras pipeline
########################################
# define network
model = Sequential()
model.add(LSTM(50, activation='tanh', input_shape=(nStepsIn, n_features)))
model.add(Dense(1))

# compile network
model.compile(optimizer='adam', loss='mse')

# fit network
model.fit(X, y, epochs=10, verbose=2, validation_split=0.2, batch_size=10)

# evaluate network
print(model.summary())

# make predictions


'''
# define model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=200, verbose=0)
# demonstrate prediction
x_input = array([[80, 85], [90, 95], [100, 105]])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)
'''
