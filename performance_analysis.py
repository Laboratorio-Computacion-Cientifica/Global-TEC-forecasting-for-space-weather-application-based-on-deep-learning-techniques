from hashlib import new
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# clculos de este archivo:
# bar plot of MSE over test set of each technique by station (18 groups of 3 bars eachone)
# bar plot of RMSE over test set of each technique by station (18 groups of 3 bars eachone)

# load df with performance of each model
pathForBaseModels = "data/performance_data/errors_for_bar_plots.csv"
pathForIncrementalModels = "data/performance_data/errors_for_bar_plots_of_incremental.csv"
df = pd.read_csv(pathForIncrementalModels)


print(df.columns)


# columns of file:
# date_range,lstm_MSE,lstm_RMSE,gru_MSE,gru_RMSE,cnn_MSE,cnn_RMSE,naive_pers_MSE,naive_pers_RMSE,naive_avg_MSE,naive_avg_RMSE

def get_data_from_file(df, dateRange):
    '''dataRange: 0: test set
                1: 24 h. ahead
                2: 07-11 Sep 2017
                3: 27-31 May 2017
                4: 26 Sep-03 Oct 2017
                5: 07-12 Nov 2017
    '''
    print(df.columns)
    selectDateRange = df.loc[df['date_range'] == dateRange]
    return selectDateRange


    
newDf = get_data_from_file(df, 1)

# bar plot of each technique by station (18 groups of 3 bars eachone)

lstmMse = newDf['lstm_MSE']
gruMse = newDf['gru_MSE']
cnnMse = newDf['cnn_MSE']
naivePersistent = newDf['naive_pers_MSE']
naiveAvg = newDf['naive_avg_MSE']
#lstmRmse = df['LSTM_RMSE']
#gruRmse = df['GRU_RMSE']
#cnnRmse = df['CNN_RMSE']

labels = ["St1","St2","St3","St4","St5","St6","St7","St8","St9","St10","St11","St12","St13","St14","St15","St16","St17","St18"]
x = np.arange(len(labels))
width = 0.1  # the width of the bars

plt.figure()
#plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.title("MSE over testset by station", fontsize=20,fontweight="bold")
plt.bar(x-width*2, lstmMse, width, label='LSTM MSE')
plt.bar(x-width, gruMse, width, label='GRU MSE')
plt.bar(x, cnnMse, width, label='CNN MSE')
plt.bar(x+width, naivePersistent, width, label='Naive Frozen Ionosphere RMSE')
plt.bar(x+width*2, naiveAvg, width, label='Naive AVG RMSE')
plt.vlines(2.5,0,40, linestyles ="dotted", colors ="g")
plt.vlines(5.5,0,40, linestyles ="dotted", colors ="g")
plt.vlines(11.5,0,40, linestyles ="dotted", colors ="g")
plt.vlines(14.5,0,40, linestyles ="dotted", colors ="g")
plt.ylabel('TECu', fontsize=16, fontweight="bold")
plt.xlabel('Stations', fontsize=20, fontweight="bold")
plt.xticks(x, labels, fontsize=20)
plt.yticks(fontsize=20)
#plt.yticks(fontweight="bold", fontsize=20)
plt.ylim(0,40)
plt.legend()


#############################
# bar plot of RMSE over test set of each technique by station (18 groups of 3 bars eachone)

lstmRMse = newDf['lstm_RMSE']
gruRMse = newDf['gru_RMSE']
cnnRMse = newDf['cnn_RMSE']
naivePersistentRMse = newDf['naive_pers_RMSE']
naiveAvgRMse = newDf['naive_avg_RMSE']

labels = ["St1","St2","St3","St4","St5","St6","St7","St8","St9","St10","St11","St12","St13","St14","St15","St16","St17","St18"]
x = np.arange(len(labels))
width = 0.2  # the width of the bars

plt.figure()
plt.rc("legend", fontsize=16)
plt.title("RMSE for each modelling technique", fontsize=20, fontweight="bold")
plt.bar(x-width*2, lstmRMse, width, label='LSTM')
plt.bar(x-width, gruRMse, width, label='GRU')
plt.bar(x, cnnRMse, width, label='CNN')
plt.bar(x+width, naivePersistentRMse, width, label='Naive Frozen Ionosphere')
plt.bar(x+width*2, naiveAvgRMse, width, label='Naive AVG 27 days')
plt.vlines(2.5,0,10, linestyles ="dotted", colors ="g")
plt.vlines(5.5,0,10, linestyles ="dotted", colors ="g")
plt.vlines(11.5,0,10, linestyles ="dotted", colors ="g")
plt.vlines(14.5,0,10, linestyles ="dotted", colors ="g")
plt.ylabel('TECu', fontsize=20, fontweight="bold")
plt.xlabel('Stations', fontsize=20, fontweight="bold")
plt.xticks(x, labels, fontsize=20)
plt.yticks(fontsize=20)
plt.ylim(0,7)
plt.legend()



'''
fig, ax = plt.subplots(1,2)
ax[0].bar(x - width/2, lstmMse, width, label='LSTM MSE')
ax[0].bar(x + width/2, gruMse, width, label='GRU MSE')
ax[0].bar(x + 3*width/2, cnnMse, width, label='CNN MSE')
ax[0].set_ylabel('MSE')
ax[0].set_xlabel('Stations')
ax[0].set_title('MSE by station')
ax[0].set_xticks(x)
ax[0].set_xticklabels(labels)
ax[0].legend()
'''

'''
ax[1].bar(x - width/2, lstmRmse, width, label='LSTM RMSE')
ax[1].bar(x + width/2, gruRmse, width, label='GRU RMSE')
ax[1].bar(x + 3*width/2, cnnRmse, width, label='CNN RMSE')
ax[1].set_ylabel('RMSE')
ax[1].set_xlabel('Stations')
ax[1].set_title('RMSE by station')
ax[1].set_xticks(x)
ax[1].set_xticklabels(labels)
ax[1].legend()
'''

#fig.tight_layout()

plt.show()


'''
# histograms of errors

fig, ax = plt.subplots(3,2)
ax[0][0].hist(lstmMse, label='LSTM MSE', align='mid')
ax[0][0].set_ylabel('MSE')
ax[0][0].set_title('LSTM MSE')
ax[0][0].legend()

ax[0][1].hist(lstmRmse, label='LSTM RMSE')
ax[0][1].set_ylabel('RMSE')
ax[0][1].set_title('LSTM RMSE')
ax[0][1].legend()

ax[1][0].hist(gruMse, label='GRU MSE')
ax[1][0].set_ylabel('MSE')
ax[1][0].set_title('GRU MSE')
ax[1][0].legend()

ax[1][1].hist(gruRmse, label='GRU RMSE')
ax[1][1].set_ylabel('RMSE')
ax[1][1].set_title('GRU RMSE')
ax[1][1].legend()

ax[2][0].hist(cnnMse, label='CNN MSE')
ax[2][0].set_ylabel('MSE')
ax[2][0].set_title('CNN MSE')
ax[2][0].legend()

ax[2][1].hist(cnnRmse, label='CNN RMSE')
ax[2][1].set_ylabel('RMSE')
ax[2][1].set_title('CNN RMSE')
ax[2][1].legend()


fig.tight_layout()

plt.show()


# histograms of errors

fig, ax = plt.subplots(1,2)
ax[0].hist(lstmMse, label='LSTM MSE')
ax[0].set_ylabel('MSE')
ax[0].set_title('LSTM MSE')
ax[0].legend()

ax[1].hist(lstmRmse, label='LSTM RMSE')
ax[1].set_ylabel('RMSE')
ax[1].set_title('LSTM RMSE')
ax[1].legend()

ax[0].hist(gruMse, label='GRU MSE')
ax[0].set_ylabel('MSE')
ax[0].set_title('GRU MSE')
ax[0].legend()

ax[1].hist(gruRmse, label='GRU RMSE')
ax[1].set_ylabel('RMSE')
ax[1].set_title('GRU RMSE')
ax[1].legend()

ax[0].hist(cnnMse, label='CNN MSE')
ax[0].set_ylabel('MSE')
ax[0].set_title('CNN MSE')
ax[0].legend()

ax[1].hist(cnnRmse, label='CNN RMSE')
ax[1].set_ylabel('RMSE')
ax[1].set_title('CNN RMSE')
ax[1].legend()


fig.tight_layout()

plt.show()
'''