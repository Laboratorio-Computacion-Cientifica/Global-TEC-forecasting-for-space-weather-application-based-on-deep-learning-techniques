import io
import os
import sys
import yaml
from numpy import hstack
from numpy import savetxt
from numpy import linspace
from datetime import datetime
from numpy import array
import pandas as pd
from selector.kp.kp_selector import select_kp_data_from_a_date_range
from selector.kp.kp_selector import transform_kp_dataframe_in_array
from selector.tec.tec_selector import build_a_tec_df_from_date_range
from selector.tec.tec_selector import transform_tec_dataframe_in_array_and_select_station
from selector.tec.tec_selector import select_data_of_a_tec_dataframe_based_on_date_range_and_station
from pre_processing_data.scalers.scaler import scale_data
from keras.models import load_model
from pre_processing_data.scalers.scaler import unscale_data
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot as plt


#!/usr/bin/env python
"""This script allow to evaluate a model in a specif data based on a specific date range, for example: 01/02/2020 using all
the data requiered, in this case: tec and kp

Steps:

1- select kp data based on date range
2- select tec data based on date range
3- cros data (tec and kp)
4- scale data based on trainset
5- select model to evaluate
6- forecast
7- plots and metrics

"""


########################################
# load data
########################################
# define input sequences for Kp index and TEC
# 1- select kp data based on date range

params = yaml.safe_load(open("params.yaml"))["evaluate_specific_date_range"]
params2 = yaml.safe_load(open("params.yaml"))["prepare"]
stationFeatures = yaml.safe_load(open("params.yaml"))["evaluate"]

# input sequence for Kp index
# path of preprocessed Kp data
kpFile = sys.argv[1]
# define init date and end date for slice the dataframe
initDate = params["dateInit"]
endDate = params["dateEnd"]
station = params2["station"]
#plotTitle = params["plotTitleForTestSet"]

# slice dataframe
with io.open(kpFile) as kpFilfeOpened:
    dfKp = select_kp_data_from_a_date_range(kpFilfeOpened, initDate, endDate)
    print("\n-------------------------------------")
    print("LOADING DATA")
    print("-------------------------------------\n")
    print("Dimension of Kp Index dataframe: {}".format(dfKp.shape))
    # transform dataframe to array
    in_seq1 = transform_kp_dataframe_in_array(dfKp)
    print("Kp Index data transformed to array dimension:  {}".format(in_seq1.shape))
    print("Kp Index data transformed to array:  {}".format(in_seq1))


# 2- select tec data based on date range

# input sequence for TEC
# path of preprocessed TEC data
tecFolder = sys.argv[2]
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
# scale and transform
########################################

print("\n-------------------------------------")
print("SCALE AND TRANSFORM DATA")
print("-------------------------------------\n")

# load train sets (tec and kp)
pathForSaveTrainSets = os.path.join(sys.argv[3], station+"_"+"trainset.txt")
dfTrainset = pd.read_csv(pathForSaveTrainSets)
tecTrainSet = dfTrainset['tec']
tecTrainSet = tecTrainSet.to_numpy()
kpTrainSet = dfTrainset['kp']
kpTrainSet = kpTrainSet.to_numpy()

# generate scalers based on trainsets of tec and kp
tecTrainSet = tecTrainSet.reshape((len(tecTrainSet), 1))
kpTrainSet = kpTrainSet.reshape((len(kpTrainSet), 1))
tecScaledData, scalerTec = scale_data(tecTrainSet,"normalization")
kpScaledData, scalerKp = scale_data(kpTrainSet,"normalization")

# scale tec and kp
in_seq1_scaled = scalerKp.transform(in_seq1.reshape(-1,1))
in_seq2_scaled = scalerTec.transform(in_seq2.reshape(-1,1))

# horizontally stack columns
in_seq1_transformed = in_seq1_scaled.reshape((len(in_seq1_scaled), 1))
in_seq2_transformed = in_seq2_scaled.reshape((len(in_seq2_scaled), 1))
dataset = hstack((in_seq2_transformed, in_seq1_transformed))
print("scaled data: \n", dataset)
print("-------------------------------------\n")


xSizeForSeelctedDate = 17
ySizeForSeelctedDate = 17 

########################################
# load model and forecast
########################################
# build inputs for the model
stepsIn = int(params2['stepsIn'])
amountOfDays =  int(dataset.shape[0]/stepsIn)
inputsForTheModel = list()
for i in range(amountOfDays-1):
    inputsForTheModel.append(dataset[i*stepsIn:i*stepsIn+stepsIn])


# load model
modelPath = sys.argv[4]
modelLstm = modelPath + "lstm_model_using_12_years_of_data.h5"
modelCnn = modelPath + "gru_model_using_12_years_of_data.h5"
modelGru = modelPath + "cnn_model_using_12_years_of_data.h5"

def forecasting_for_models(path, inputsForTheModel):
    model = load_model(path)
    # make predictions
    predictions = list()

    for i in range(len(inputsForTheModel)):
        X = inputsForTheModel[i]
        X = X.reshape(1, X.shape[0], X.shape[1])
        yhat = model.predict(X, verbose=0)
        yhat = unscale_data(yhat[0].reshape(-1,1),scalerTec)
        predictions.append(yhat.tolist())
    # transform the predictions to one list
    predictedValues = list()
    for pred in predictions:
        for j in pred:
            predictedValues.append(j)
    return predictedValues
        
predictedValuesLstm = forecasting_for_models(modelLstm, inputsForTheModel)
predictedValuesGru = forecasting_for_models(modelGru, inputsForTheModel)
predictedValuesCnn = forecasting_for_models(modelCnn, inputsForTheModel)

predictedValues = list()

for i in range(len(predictedValuesLstm)):
    avgOfElements = (float(predictedValuesLstm[i][0])+float(predictedValuesGru[i][0])+float(predictedValuesCnn[i][0]))/3.
    predictedValues.append(avgOfElements)

########################################
# metrics
########################################

#print("Real values {} (len: {})".format(in_seq2[12:],len(in_seq2[12:])))
#print("Predictions values {} (len: {}): ".format(predictions, len(predictions)))

techniquedUsedForTitle = sys.argv[5]
techniquedUsedForFileName = sys.argv[5]
print("techniquedUsedForFileName", techniquedUsedForFileName)


# errors
print("\n errors for "+techniquedUsedForTitle+": ")
mse = mean_squared_error(in_seq2[12:],predictedValues)
rmse = sqrt(mse)
print("MSE  = ", mse)
print("RMSE  = ", rmse)

# plots
dateForTitle = str(endDate)
dateForTitle = str(dateForTitle[0:4]) + '/' + str(dateForTitle[4:6])  + '/' + str(dateForTitle[6:])


def  define_xticks_labels(amountOfDays, isTestSet, initData="nothing"):
    listOfValues = list()
    if(isTestSet):
        listOfValues = ["19/Nov","29/Nov","09/Dec","19/Dec","29/Dec"]
        title1 = techniquedUsedForFileName.upper() + " method versus actual data on the test set (Nov-Dec 2016)"
    else:
        '''
        for i in range(amountOfDays):
            for j in range(0,23,12):
                listOfValues.append(j)
        '''
        if(str(initDate) == "20171106"):
            listOfValues = ["07/Nov","08/Nov","09/Nov","10/Nov","11/Nov","12/Nov"]
            title1 = techniquedUsedForFileName.upper() + " method versus actual data on date range 07/Nov-12/Nov of 2017"

        if(str(initDate) == "20170925"):
            listOfValues = ["26/Sep","27/Sep","28/Sep","29/Sep","30/Sep","01/Oct","02/Oct","03/Oct"]
            title1 = techniquedUsedForFileName.upper() + " method versus actual data on date range 26/Sep-03/Oct of 2017"
        if(str(initDate) == "20170526"):
            listOfValues = ["27/May","28/May","29/May","30/May","31/May"]
            title1 = techniquedUsedForFileName.upper() + " method versus actual data on date range 27/May-31/May of 2017"
        if(str(initDate) == "20170906"):
            listOfValues = ["07/Sep","08/Sep","09/Sep","10/Sep","11/Sep"]
            title1 = techniquedUsedForFileName.upper() + " method versus actual data on date range 07/Sep-11/Sep of 2017"
      
    return listOfValues, title1


isTestSet = params["isTestSet"]

if(isTestSet):
    fig, axs = plt.subplots(1, 1, figsize=(30, 15), tight_layout=False)
else:
    fig, axs = plt.subplots(1, 1, figsize=(30, 15), tight_layout=False)


fontSizeTitle = 24
fontSizeAxis = 24

#plt.figure(figsize=(5, 3), dpi=200, constrained_layout=True)
#plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.4f}'))
#plt.title(station, fontsize=20, fontweight='bold')
#axs.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
axs.yaxis.get_offset_text().set_fontsize(20)
if(isTestSet):
    axs.tick_params(axis='x', labelsize=30)
    axs.tick_params(axis='y', labelsize=30)
else:
    axs.tick_params(axis='x', labelsize=xSizeForSeelctedDate)
    axs.tick_params(axis='y', labelsize=ySizeForSeelctedDate)
#axs.set_xlim(0, 21)
amountOfDaysOfRange = params['amountOfDays']
if(isTestSet):
    xticks = [i for i in range(0,amountOfDaysOfRange*12+2,120)]
else:
    xticks = [i for i in range(0,amountOfDaysOfRange*12+2,12)]
xtickLabels, plotTitle = define_xticks_labels(amountOfDays, isTestSet, initDate)
axs.set_xticks(xticks)
for tick in axs.xaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')
for tick in axs.yaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')
axs.set_xticklabels(xtickLabels)
import numpy as np
domain = np.arange(0,amountOfDaysOfRange*12)
axs.plot(domain, in_seq2[12:],linewidth = '5')
axs.plot(domain, predictedValues, linewidth = '5')
axs.tick_params(axis='y', labelsize=fontSizeAxis)
axs.tick_params(axis='x', labelsize=fontSizeAxis)
axs.set_xlabel("Day",fontsize=fontSizeAxis,fontweight='bold')
axs.margins(0)
axs.grid()
axs.set_title(plotTitle, fontsize=20,fontweight='bold')
fig.savefig('data/models/evaluation_model_specific_range_'+str(initDate)+'_'+str(endDate)+techniquedUsedForFileName+'.png')
plt.show()




############################
# plot with sym-h
############################

if(not isTestSet):

    pathSymH = params['pathSymH']
    df = pd.read_csv(pathSymH)

    ddI = str(initDate)[0:4]
    mmI = str(initDate)[4:6]
    yyI = str(initDate)[6:]
    yyImmIddI = ddI+'-'+mmI+'-'+str(yyI)

    ddE = str(endDate)[0:4]
    mmE = str(endDate)[4:6]
    yyE = str(endDate)[6:]
    yyEmmEddE = ddE+'-'+mmE+'-'+yyE

    dfAux2 = df.loc[(df.timestamp > yyImmIddI) & (df.timestamp <= yyEmmEddE)]
    #take data with 2 h resolution
    dfAux = pd.DataFrame(columns=dfAux2.columns)
    for i in range(0,dfAux2.shape[0],120):
        dfAux = dfAux.append(dfAux2.iloc[i])


    if(isTestSet):
        fig, axs = plt.subplots(2, 1, figsize=(30, 15), tight_layout=False)
    else:
        fig, axs = plt.subplots(2, 1, figsize=(30, 15), tight_layout=False,sharex=True)        

    #plt.figure(figsize=(5, 3), dpi=200, constrained_layout=True)
    #plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.4f}'))
    #plt.title(station, fontsize=20, fontweight='bold')
    #axs[0].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    #axs[0].xaxis.get_offset_text().set_fontsize(25)
    #axs[0].yaxis.get_offset_text().set_fontsize(20)
    #axs[1].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    #axs[1].xaxis.get_offset_text().set_fontsize(25)
    #axs[1].yaxis.get_offset_text().set_fontsize(25)    
    axs[0].tick_params(axis='x', labelsize=xSizeForSeelctedDate)
    axs[0].tick_params(axis='y', labelsize=ySizeForSeelctedDate)
    axs[1].tick_params(axis='x', labelsize=xSizeForSeelctedDate)
    axs[1].tick_params(axis='y', labelsize=ySizeForSeelctedDate)
    #axs.set_xlim(0, 21)
    amountOfDaysOfRange = params['amountOfDays']
    if(isTestSet):
        xticks = [i for i in range(0,amountOfDaysOfRange*12+2,120)]
    else:
        xticks = [i for i in range(0,amountOfDaysOfRange*12+2,12)]
    xtickLabels, plotTitle = define_xticks_labels(amountOfDaysOfRange, isTestSet, initDate)
    import numpy as np
    domain = np.arange(0,amountOfDaysOfRange*12)
    axs[0].set_title(plotTitle, fontsize=fontSizeTitle,fontweight='bold')
    axs[0].plot(domain, in_seq2[12:],linewidth = '5')
    axs[0].plot(domain, predictedValues, linewidth = '5')
    axs[0].set_ylabel("TEC [TECu]",fontsize=fontSizeAxis,fontweight='bold')    
    axs[0].tick_params(axis='y', labelsize=fontSizeAxis)
    axs[0].margins(0)
    axs[0].grid()
    #axs[0].set_title(plotTitle, fontsize=20,fontweight='bold')
    axs[1].plot(domain, dfAux["SYM-H"], color="black", linewidth = '5')
    axs[1].set_xlabel("Day",fontsize=fontSizeAxis,fontweight='bold')
    axs[1].set_ylabel("Sym-H [nT]",fontsize=fontSizeAxis,fontweight='bold')    
    axs[1].set_xticks(xticks)
    axs[1].set_xticklabels(xtickLabels, fontsize=fontSizeAxis)
    axs[1].tick_params(axis='y', labelsize=fontSizeAxis)
    for tick in axs[0].xaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for tick in axs[0].yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')    
    for tick in axs[1].xaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for tick in axs[1].yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    axs[1].margins(0)


    for tick in axs[0].xaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for tick in axs[0].yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')    
    for tick in axs[1].xaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for tick in axs[1].yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')    

    fig.savefig('data/models/evaluation_model_specific_range_'+str(initDate)+'_'+str(endDate)+techniquedUsedForFileName+'with_symh.png')
    plt.subplots_adjust(hspace=.0)
    plt.show()