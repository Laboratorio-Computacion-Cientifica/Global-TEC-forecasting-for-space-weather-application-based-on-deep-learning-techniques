import io
import os
import sys
from matplotlib import pyplot as plt
import pandas as pd


path0 = sys.argv[1]
df1 = pd.read_csv(path0)
path1 = sys.argv[2]
df2 = pd.read_csv(path1)

df1 = df1.append(df2, ignore_index=True)

plt.hist(df1["tec"], label="traiset")

plt.title("Station 1 - Data from 2005/01/01 to 2016/12/31", fontweight="bold", fontsize=14)
plt.xlabel("$\\bf{TEC (TECu)}$", fontsize=12)
plt.ylabel("$\\bf{Frequency}$", fontsize=12)
plt.margins(0)
plt.grid()
plt.show()
