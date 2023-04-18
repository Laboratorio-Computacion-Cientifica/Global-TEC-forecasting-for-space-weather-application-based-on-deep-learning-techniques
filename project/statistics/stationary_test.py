import pandas as pd
from matplotlib import pyplot as plt
from pandas import read_csv
from statsmodels.tsa.stattools import adfuller



'''
source: https://machinelearningmastery.com/time-series-data-stationary-python/
When a time series is stationary, it can be easier to model. Statistical modeling methods assume or require the time series to be stationary to be effective.

The temporal structure adds an order to the observations. This imposed order means that important assumptions about the consistency of those observations needs to be handled specifically.
For example, when modeling, there are assumptions that the summary statistics of observations are consistent. In time series terminology, we refer to this expectation as the time series being stationary.
These assumptions can be easily violated in time series by the addition of a trend, seasonality, and other time-dependent structures.

Null Hypothesis (H0): If failed to be rejected, it suggests the time series has a unit root, meaning it is non-stationary. It has some time dependent structure.
Alternate Hypothesis (H1): The null hypothesis is rejected; it suggests the time series does not have a unit root, meaning it is stationary. It does not have time-dependent structure.

p-value > 0.05: Fail to reject the null hypothesis (H0), the data has a unit root and is non-stationary.
p-value <= 0.05: Reject the null hypothesis (H0), the data does not have a unit root and is stationary.
'''



df = pd.read_csv('/home/jorge/Desktop/Doctorado/work_with_italy/repo_ingv_tec_prediction/data/prepared/station1_trainset.txt')

tec = df['tec']
print(tec.shape)
tec.plot.hist()

plt.show()


result = adfuller(tec)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
