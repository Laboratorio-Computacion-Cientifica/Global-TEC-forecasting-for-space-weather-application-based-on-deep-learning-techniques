import io
import os
import sys
import yaml
import pickle
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.initializers import Identity
from keras.initializers import GlorotNormal
from keras import callbacks
import numpy as np
from matplotlib import pyplot as plt
from keras.regularizers import l1_l2

params = yaml.safe_load(open("params.yaml"))["train"]
params2 = yaml.safe_load(open("params.yaml"))["prepare"]
params3 = yaml.safe_load(open("params.yaml"))["evaluate"]

# load filenames of X and y
xFile = sys.argv[1]
yFile = sys.argv[2]

# load X
X = np.loadtxt(xFile)
# it was saved as 2d array, it need to be reshaped to 3d
_nFeatures = params["nFeatures"]
X = X.reshape(X.shape[0],X.shape[1]//2,_nFeatures)
# load y
y = np.loadtxt(yFile)

print(X.shape)
print(y.shape)

# parameters
_activation1 = params["activation1"]
_activation2 = params["activation2"]
_optimizer = params["optimizer"]
_loss = params["loss"]
_verbose = params["verbose"]
_validation = params["validation"]
_epochs = params["epochs"]
_batch = params["batch"]
_stepsOut = params2["stepsOut"]
# define model
model = Sequential()
n_features = X.shape[2]
# define model
model = Sequential()
model.add(Conv1D(32, 2, activation='relu', input_shape=(X.shape[1], _nFeatures)))
model.add(MaxPooling1D())
model.add(Flatten())
#model.add(Dense(50, activation='relu'))
model.add(Dense(_stepsOut))
model.compile(optimizer='adam', loss='mse')
# fit model

# fit model
historyFilename='data/models/history_log_'+params2['station']+'_cnn.csv'
history_logger= callbacks.CSVLogger(historyFilename, separator=",", append=True)
history = model.fit(X, y, epochs=_epochs, callbacks=[history_logger], validation_split=_validation,  batch_size=_batch, verbose=_verbose)

print(len(history.history['loss']))

# summarize history for loss
plt.rcParams.update({'font.size': 20, 'font.weight':'bold'})
plt.figure(figsize=(30, 20), dpi=80)
plt.title("Loss function on "+params2['station']+" - lat: "+str(params3['lat'])+", long: "+str(params3['long']), fontweight='bold', fontsize=30)

xDomain = range(1,21,1)
plt.plot(xDomain, history.history['loss'])
plt.plot(xDomain, history.history['val_loss'])
plt.xticks(np.arange(min(xDomain), max(xDomain)+1, 1))

plt.ylabel('loss', fontweight='bold', fontsize=20)
plt.xlabel('epoch', fontweight='bold', fontsize=20)
plt.legend(['train', 'val'], loc='best')
plt.savefig('data/models/train_val_loss_cnn.png')

plt.show()



initDate = params2["dateInit"]
endDate = params2["dateEnd"]
yearInit = int(str(initDate)[0:4])
yearEnd = int(str(endDate)[0:4])
amountOfYearusedForTrain = yearEnd-yearInit+1
model.save('data/models/cnn_model_using_'+str(amountOfYearusedForTrain)+'_years_of_data.h5')

#output = sys.argv[3]
#model.save(output)

# https://machinelearningmastery.com/diagnose-overfitting-underfitting-lstm-models/



# dvc run -n train -p train.activation -p train.optimizer -p train.loss -p train.nFeatures -p train.verbose -p train.validation -p train.epochs -d data/prepared/station1_x_trainset_scaled_and_transformed.txt -d data/prepared/station1_y_trainset_scaled_and_transformed.txt -o model.pkl python project/train.py data/prepared/station1_x_trainset_scaled_and_transformed.txt data/prepared/station1_y_trainset_scaled_and_transformed.txt model.pkl
