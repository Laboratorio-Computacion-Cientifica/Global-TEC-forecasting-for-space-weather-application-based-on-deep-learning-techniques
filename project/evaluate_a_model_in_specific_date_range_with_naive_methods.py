import io
import os
import sys
from turtle import color
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
#plotTitle1 = params["plotTitle1"]
#plotTitle2 = params["plotTitle2"]

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

xSizeForSeelctedDate = 17
ySizeForSeelctedDate = 17

xSizeForSeelctedDateForGrid = 14
ySizeForSeelctedDateForGrid = 14

fontSizeTitleForGridPlot = 12
fontSizeAxisForGridPlot = 10

########################################
# persistent method
########################################

def define_xticks_labels(amountOfDays, isTestSet, initDate="nothing"):
    listOfValues = list()
    if(isTestSet):
        listOfValues = ["19/Nov","29/Nov","09/Dec","19/Dec","29/Dec"]
        title1 = "Frozen ionosphere method versus actual data on the test set (Nov-Dec 2016)"
        title2 = "27-days-avg naive method versus actual data on the test set (Nov-Dec 2016)"
    else:
        if(str(initDate) == "20171106"):
            listOfValues = ["07/Nov","08/Nov","09/Nov","10/Nov","11/Nov","12/Nov"]
            title1 = "Frozen ionosphere method versus actual data on date range 07/Nov-12/Nov of 2017"
            title2 = "27-days-avg naive method versus actual data on date range 07/Nov-12/Nov of 2017"
        if(str(initDate) == "20170925"):
            listOfValues = ["26/Sep","27/Sep","28/Sep","29/Sep","30/Sep","01/Oct","02/Oct","03/Oct"]
            title1 = "Frozen ionosphere method versus actual data on date range 26/Sep-03/Oct of 2017"
            title2 = "27-days-avg naive method versus actual data on date range 26/Sep-03/Oct of 2017"
        if(str(initDate) == "20170526"):
            listOfValues = ["27/May","28/May","29/May","30/May","31/May"]
            title1 = "Frozen ionosphere method versus actual data on date range 27/May-31/May of 2017"
            title2 = "27-days-avg naive method versus actual data on date range 27/May-31/May of 2017"
        if(str(initDate) == "20170906"):
            listOfValues = ["07/Sep","08/Sep","09/Sep","10/Sep","11/Sep"]
            title1 = "Frozen ionosphere method versus actual data on date range 07/Sep-11/Sep of 2017"
            title2 = "27-days-avg naive method versus actual data on date range 07/Sep-11/Sep of 2017"
      
    return listOfValues, title1, title2

predictedValues = in_seq2[:-12]

isTestSet = params["isTestSet"]
amountOfDaysOfRange = params['amountOfDays']

if(isTestSet):
    fig, axs = plt.subplots(1, 1, figsize=(30, 15), tight_layout=False)
    fontSizeTitle = 20
    fontSizeAxis = 20
    xticks = [i for i in range(0,amountOfDaysOfRange*12+2,120)]
    axs.tick_params(axis='x', labelsize=20)
    axs.tick_params(axis='y', labelsize=20)
    xtickLabels, plotTitle1, plotTitle2 = define_xticks_labels(amountOfDaysOfRange, isTestSet, initDate)
    axs.set_title(plotTitle1, fontsize=fontSizeTitle,fontweight='bold')
    axs.set_xlabel("Day",fontsize=fontSizeAxis,fontweight='bold')
    axs.set_ylabel("TEC [TECu]",fontsize=fontSizeAxis,fontweight='bold')   
else:
    fig, axs = plt.subplots(1, 1, figsize=(6, 3), tight_layout=False)
    xticks = [i for i in range(0,amountOfDaysOfRange*12+2,12)]
    #fontSizeTitle = fontSizeTitleForGridPlot
    #fontSizeAxis = fontSizeAxisForGridPlot

#fig, axs = plt.subplots(1, 1, figsize=(6, 3), tight_layout=False)
#axs.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
#axs.yaxis.get_offset_text().set_fontsize(20)



xtickLabels, plotTitle1, plotTitle2 = define_xticks_labels(amountOfDaysOfRange, isTestSet, initDate)
axs.set_xticks(xticks)
axs.set_xticklabels(xtickLabels)
for tick in axs.xaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')
for tick in axs.yaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')
import numpy as np
domain = np.arange(0,amountOfDaysOfRange*12)
axs.plot(domain, in_seq2[12:],linewidth = '5')
axs.plot(domain, predictedValues, '--', linewidth = '3', c= ((0,0.7,0,1)))
axs.margins(0)
axs.grid()
fig.savefig('data/models/evaluation_model_specific_range_'+str(initDate)+'_'+str(endDate)+'naive_persistent'+'.png')
plt.show()

# errors
print("\n errors for naive persistent in selected range")
mse = mean_squared_error(in_seq2[12:],predictedValues)
rmse = sqrt(mse)
print("MSE  = ", mse)
print("RMSE  = ", rmse)


########################################
# plot with sym-h
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

    fig, axs = plt.subplots(2, 1, figsize=(30, 15), tight_layout=False,sharex=True)
    fontSizeTitle = 24
    fontSizeAxis = 24

    #plt.figure(figsize=(5, 3), dpi=200, constrained_layout=True)
    #plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.4f}'))
    #plt.title(station, fontsize=20, fontweight='bold')
    #axs[0].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    #axs[0].yaxis.get_offset_text().set_fontsize(20)
    #axs[1].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    #axs[1].yaxis.get_offset_text().set_fontsize(20)

    #axs[0].tick_params(axis='x', labelsize=xSizeForSeelctedDate)
    #axs[0].tick_params(axis='y', labelsize=ySizeForSeelctedDate)
    #axs[1].tick_params(axis='x', labelsize=xSizeForSeelctedDate)
    #axs[1].tick_params(axis='y', labelsize=ySizeForSeelctedDate)
    #axs.set_xlim(0, 21)
    amountOfDaysOfRange = params['amountOfDays']
    if(isTestSet):
        xticks = [i for i in range(0,amountOfDaysOfRange*12+2,120)]
    else:
        xticks = [i for i in range(0,amountOfDaysOfRange*12+2,12)]
    xtickLabels, plotTitle1, plotTitle2 = define_xticks_labels(amountOfDaysOfRange, isTestSet, initDate)

    import numpy as np
    domain = np.arange(0,amountOfDaysOfRange*12)
    axs[0].plot(domain, in_seq2[12:],linewidth = '5')
    axs[0].plot(domain, predictedValues, '--', linewidth = '3', c= ((0,0.7,0,1)))
    axs[0].margins(0)
    axs[0].grid()
    axs[0].set_title(plotTitle1, fontsize=fontSizeTitle,fontweight='bold')
    #axs[0].set_xlabel("UT",fontsize=fontSizeAxis,fontweight='bold')
    axs[0].set_ylabel("TEC [TECu]",fontsize=fontSizeAxis,fontweight='bold')
    axs[0].tick_params(axis='y', labelsize=fontSizeAxis)

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
    fig.savefig('data/models/evaluation_model_specific_range_'+str(initDate)+'_'+str(endDate)+'naive_persistent'+'with_symh.png')
    plt.subplots_adjust(hspace=.0)
    plt.show()
    print(len(domain))
    print(dfAux.shape)

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
for i in range(amountOfDaysOfRange+1):
    auxList = avg_point_to_point(amountOfDataPerDay, amountOfDays, in_seq2_avg[i*amountOfDataPerDay:])
    naiveDatasetAvg27Days.append(auxList)


# transform list of list to list
predictedValues = list()
for l in naiveDatasetAvg27Days:
    for e in l:
        predictedValues.append(e)

if(isTestSet):
    fig, axs = plt.subplots(1, 1, figsize=(30, 15), tight_layout=False)
    fontSizeTitle = 20
    fontSizeAxis = 20
    xticks = [i for i in range(0,amountOfDaysOfRange*12+2,120)]
    axs.tick_params(axis='x', labelsize=20)
    axs.tick_params(axis='y', labelsize=20)    
    axs.set_xlabel("Day",fontsize=fontSizeAxis,fontweight='bold')
    axs.set_ylabel("TEC [TECu]",fontsize=fontSizeAxis,fontweight='bold')   
else:
    fig, axs = plt.subplots(1, 1, figsize=(6, 3), tight_layout=False)
    xticks = [i for i in range(0,amountOfDaysOfRange*12+2,12)]


xtickLabels, plotTitle1, plotTitle2 = define_xticks_labels(amountOfDaysOfRange, isTestSet, initDate)
axs.set_xticks(xticks)
axs.set_xticklabels(xtickLabels)
for tick in axs.xaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')
for tick in axs.yaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')
domain = np.arange(0,amountOfDaysOfRange*12)
#axs.set_title(plotTitle2, fontsize=fontSizeTitle,fontweight='bold')
axs.plot(domain, in_seq2[12:],linewidth = '5')
axs.plot(domain, predictedValues[12:], '--', linewidth = '3', c= ((0,0.7,0,1)))
axs.margins(0)
axs.grid()
fig.savefig('data/models/evaluation_model_specific_range_'+str(initDate)+'_'+str(endDate)+'naive_avg'+'.png')
plt.show()

# errors
print("\n errors for naive avg 27 days in selected range")
mse = mean_squared_error(in_seq2[12:],predictedValues[12:])
rmse = sqrt(mse)
print("MSE  = ", mse)
print("RMSE  = ", rmse)

########################################
# plot avg with sym-h
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


    fig, axs = plt.subplots(2, 1, figsize=(30, 15), tight_layout=False,sharex=True)
    fontSizeTitle = 24
    fontSizeAxis = 24       

    #plt.figure(figsize=(5, 3), dpi=200, constrained_layout=True)
    #plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.4f}'))
    #plt.title(station, fontsize=20, fontweight='bold')
    #axs[0].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    #axs[0].yaxis.get_offset_text().set_fontsize(20)
    #axs[1].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    #axs[1].yaxis.get_offset_text().set_fontsize(20)

    #axs[0].tick_params(axis='x', labelsize=xSizeForSeelctedDate)
    #axs[0].tick_params(axis='y', labelsize=ySizeForSeelctedDate)
    #axs[1].tick_params(axis='x', labelsize=xSizeForSeelctedDate)
    #axs[1].tick_params(axis='y', labelsize=ySizeForSeelctedDate)
    #axs.set_xlim(0, 21)
    amountOfDaysOfRange = params['amountOfDays']
    if(isTestSet):
        xticks = [i for i in range(0,amountOfDaysOfRange*12+2,120)]
    else:
        xticks = [i for i in range(0,amountOfDaysOfRange*12+2,12)]
    xtickLabels, plotTitle1, plotTitle2 = define_xticks_labels(amountOfDays, isTestSet, initDate)
    #axs[0].set_xticks(xticks)
    #axs[0].set_xticklabels(xtickLabels)

    domain = np.arange(0,amountOfDaysOfRange*12)
    axs[0].plot(domain, in_seq2[12:],linewidth = '5')
    axs[0].plot(domain, predictedValues[12:], '--', linewidth = '3', c= ((0,0.7,0,1)))
    axs[0].tick_params(axis='y', labelsize=fontSizeAxis)
    axs[0].margins(0)
    axs[0].grid()
    axs[0].set_title(plotTitle2, fontsize=fontSizeTitle,fontweight='bold')
    #axs[0].set_xlabel("UT",fontsize=fontSizeAxis,fontweight='bold')
    axs[0].set_ylabel("TEC [TECu]",fontsize=fontSizeAxis,fontweight='bold')

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
    plt.subplots_adjust(hspace=.0)
    fig.savefig('data/models/evaluation_model_specific_range_'+str(initDate)+'_'+str(endDate)+'naive_avg_with_symh'+'.png')
    plt.show()