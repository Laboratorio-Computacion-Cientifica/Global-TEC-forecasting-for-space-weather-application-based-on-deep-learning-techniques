import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("/home/jorge/Downloads/final_symh_2017_to_2018.csv")

dfAux = df.loc[(df.timestamp >= '2017-09-07') & (df.timestamp <= '2017-09-11')]

plt.plot(dfAux["SYM-H"])
plt.margins(0)
plt.show()