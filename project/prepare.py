import io
import os
import sys
import yaml
from numpy import hstack
from numpy import savetxt
import pandas as pd 
from selector.kp.kp_selector import select_kp_data_from_a_date_range
from selector.tec.tec_selector import build_a_tec_df_from_several_files
from selector.tec.tec_selector import build_a_tec_df_from_date_range
from selector.kp.kp_selector import transform_kp_dataframe_in_array
from selector.tec.tec_selector import transform_tec_dataframe_in_array_and_select_station
from pre_processing_data.splitters.split_sequence import train_test_split
from pre_processing_data.scalers.scaler import scale_data
from pre_processing_data.splitters.multivariate_splitters import sliding_window


params = yaml.safe_load(open("params.yaml"))["prepare"]

########################################
# load data
########################################
# define input sequences for Kp index and TEC


# input sequence for Kp index
# path of preprocessed Kp data
kpFile = sys.argv[1]
# define init date and end date for slice the dataframe
initDate = params["dateInit"]
endDate = params["dateEnd"]
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
in_seq2 = transform_tec_dataframe_in_array_and_select_station(dfTec, params["station"])
print("TEC data transformed to array dimension:  {}".format(in_seq2.shape))
print("TEC data transformed to array:  {}\n".format(in_seq2))
print("-------------------------------------\n")


########################################
# split data in train, test and validation subsets and scale
########################################
# split data
n_test = params["nTest"]
kpTrainSet, kpTestSet, amount = train_test_split(in_seq1[params["shift"]:], n_test, params["stepsIn"])
tecTrainSet, tecTestSet, amount = train_test_split(in_seq2[params["shift"]:], n_test, params["stepsIn"])

print("\n-------------------------------------")
print("SPLITING DATA IN TRAINSET AND TESTSET.")
print("-------------------------------------\n")
print("Percentage of testset : {}".format(n_test))
print("TEC train and test sets dimension: {} (DOYs {}-{}) {} (DOYs until {})".format(len(tecTrainSet),dfTec.iloc[0]['DATE'], dfTec.iloc[-amount]['DATE'], len(tecTestSet),dfTec.iloc[-1]['DATE']))
print("Kp Index train and test sets dimension: {} {}".format(len(kpTrainSet), len(kpTestSet)))
print("\n-------------------------------------\n")


########################################
# save trainset and testset
########################################
pathForSaveTrainSets = os.path.join(sys.argv[3], params["station"]+"_"+"trainset.txt")
pathForSaveTestSets = os.path.join(sys.argv[3], params["station"]+"_"+"testset.txt")
print("\n-------------------------------------")
print("SAVING TRAINSET AND TESTSET")
print("-------------------------------------\n")
print("TEC and Kp Index trainset: {} ".format(pathForSaveTrainSets))
print("TEC and Kp Index testset: {} ".format(pathForSaveTestSets))
print("\n-------------------------------------\n")

# save trainset
# convert to [rows, columns] structure
tecTrainSet = tecTrainSet.reshape((len(tecTrainSet), 1))
kpTrainSet = kpTrainSet.reshape((len(kpTrainSet), 1))
# horizontally stack columns

dataset = hstack((tecTrainSet, kpTrainSet))
pd.DataFrame(dataset, columns=["tec","kp"]).to_csv(pathForSaveTrainSets)

# save testset
# convert to [rows, columns] structure
tecTestSet = tecTestSet.reshape((len(tecTestSet), 1))
kpTestSet = kpTestSet.reshape((len(kpTestSet), 1))
# horizontally stack columns
dataset = hstack((tecTestSet, kpTestSet))
# save data
pd.DataFrame(dataset, columns=["tec","kp"]).to_csv(pathForSaveTestSets)


########################################
# scale, transform and save trainset
########################################
# define destiny
pathForSaveTrainSetsScaledOfX = os.path.join(sys.argv[3], params["station"]+"_"+"x_trainset_scaled_and_transformed.txt")
pathForSaveTrainSetsScaledOfY = os.path.join(sys.argv[3], params["station"]+"_"+"y_trainset_scaled_and_transformed.txt")

# scale data
tecScaledData, scaler = scale_data(tecTrainSet,"normalization")
kpScaledData, scaler = scale_data(kpTrainSet,"normalization")

# transform data
# convert to [rows, columns] structure
in_seq1 = tecScaledData.reshape((len(tecScaledData), 1))
in_seq2 = kpScaledData.reshape((len(kpScaledData), 1))
out_seq = tecScaledData.reshape((len(tecScaledData), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))
# choose a number of time steps
nStepsIn, nStepsOut = params["stepsIn"], params["stepsOut"]
# convert into input/output
X, y = sliding_window(dataset, nStepsIn, nStepsOut)
# save data
# source: https://www.geeksforgeeks.org/how-to-load-and-save-3d-numpy-array-to-file-using-savetxt-and-loadtxt-functions/

print("\n-------------------------------------")
print("SAVING TRANSFORMED DATA")
print("-------------------------------------\n")
print("X shape: {}".format(X.shape))
print("y shape: {}".format(y.shape))


# reshape numpy array to 2d to save it
X = X.reshape(X.shape[0], -1)
savetxt(pathForSaveTrainSetsScaledOfX, X)
savetxt(pathForSaveTrainSetsScaledOfY, y)

print("X reshaped for save it: {}".format(X.shape))
print("y reshaped for save it: {}".format(y.shape))


# tips for load numpy array saved
# b = gfg.loadtxt(<filename>)
# c = b.reshape(b.shape[0],b.shape[1]//2,2)

