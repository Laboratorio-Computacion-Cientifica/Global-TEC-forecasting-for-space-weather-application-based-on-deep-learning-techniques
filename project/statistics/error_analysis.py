import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('/home/jorge/Desktop/Doctorado/work_with_italy/stations_analysis/plots_and_errors_5_years/errors_statistics.txt')

shifted_rmse_one_day = df['shifted_rmse_one_day']
shifted_rmse_testset = df['shifted_rmse_testset']
not_shifted_rmse_one_day = df['not_shifted_rmse_one_day']
not_shifted_rmse_testset = df['not_shifted_rmse_testset']

plt.subplot(121)
plt.gca().set_title('Error of one day forecast with predictions shifted one step window', fontweight='bold')
shifted_rmse_one_day.hist()
plt.ylabel('Error', fontweight='bold')
plt.subplot(122)
plt.gca().set_title('Error of one day forecast with predictions not shifted one step window', fontweight='bold')
not_shifted_rmse_one_day.hist()
plt.ylabel('Error', fontweight='bold')

plt.show()



plt.subplot(121)
plt.gca().set_title('Error of forecast over testset with predictions shifted one step window', fontweight='bold')
plt.title('Error of one day forecast with predictions shifted one step window', fontweight='bold')
shifted_rmse_testset.hist()
plt.ylabel('Error', fontweight='bold')
plt.subplot(122)
plt.gca().set_title('Error of forecast over testset with predictions not shifted one step window', fontweight='bold')
not_shifted_rmse_testset.hist()
plt.ylabel('Error', fontweight='bold')

plt.show()



#shifted_rmse_one_day.barplot(fontsize=6, label= 'Predictions shifted')
import numpy as np
x = np.linspace(0,len(shifted_rmse_one_day),len(shifted_rmse_one_day))
#plt.errorbar(x, shifted_rmse_one_day, yerr=[1,2,3,4,5,5,6,7,8,9,10,3,4,5,5,6,7,8])
#plt.bar(x, shifted_rmse_one_day)
mask1 = shifted_rmse_one_day <1.5 
mask2 = shifted_rmse_one_day <2.5
mask3 = shifted_rmse_one_day >=2.5

#plt.bar(x[mask2], shifted_rmse_one_day[mask2], color = '', align='center', width=.8)
#plt.bar(x[mask1], shifted_rmse_one_day[mask1], color = 'green', align='center', width=.8)
#plt.bar(x[mask3], shifted_rmse_one_day[mask3], color = 'red', align='center', width=.8)

height = shifted_rmse_one_day
bars = ['ST1\n(-85,-120)','ST2\n(-85,0)','ST3\n(-85,120)','ST4\n(-50,-120)','ST5\n(-50,0)','ST6\n(-50,120)','ST7\n(-20,-120)','ST8\n(-20,0)','ST9\n(-20,120)','ST10\n(20,-120)','ST11\n(20,0)','ST12\n(20,120)','ST13\n(50,-120)','ST14\n(50,0)','ST15\n(50,120)','ST16\n(85,-120)','ST17\n(85,0)','ST18\n(85,120)']
# Choose the width of each bar and their positions
width = [0.5]*18
x_pos = range(18)
plt.bar(x_pos, height, width=width)
plt.xticks(x_pos, bars, fontsize=11, fontweight='bold')
plt.title('Error vs. (Latitude and Longitude)', fontweight='bold', fontsize=14)
#plt.xticks(range(18),['ST1\n(-85,-120)','ST2\n(-85,0)','ST3\n(-85,120)','ST4\n(-50,-120)','ST5\n(-50,0)','ST6\n(-50,120)','ST7\n(-20,-120)','ST8\n(-20,0)','ST9\n(-20,120)','ST10\n(20,-120)','ST11\n(20,0)','ST12\n(20,-120)','ST13\n(50,-120)','ST14\n(50,0)','ST15\n(50,120)','ST16\n(85,-120)','ST17\n(85,0)','ST18\n(85,120)'], fontweight='bold', fontsize=10)
plt.xlabel('Station Nº(Lat., Long.)', fontweight='bold', fontsize=10)
plt.ylabel('Error', fontweight='bold', fontsize=10)
plt.vlines(x=5.5,ymin=0,ymax=4.5,linestyles='dashed',colors='blue')
plt.vlines(x=12.5,ymin=0,ymax=4.5,linestyles='dashed',colors='blue')
plt.legend()
plt.show()


####################
#AVG by latitude
###################

print(shifted_rmse_one_day.iloc[0:3])
lat1 = shifted_rmse_one_day.iloc[0:3].mean()
lat2 = shifted_rmse_one_day.iloc[3:6].mean()
lat3 = shifted_rmse_one_day.iloc[6:9].mean()
lat4 = shifted_rmse_one_day.iloc[9:12].mean()
lat5 = shifted_rmse_one_day.iloc[12:15].mean()
lat6 = shifted_rmse_one_day.iloc[15:].mean()
latsAvg = [lat1, lat2, lat3, lat4, lat5, lat6]


height = latsAvg
bars = ['Lat -85','Lat -50','Lat -20','Lat 20','Lat 50','Lat 85']
# Choose the width of each bar and their positions
width = [0.5]*6
x_pos = range(6)
plt.bar(x_pos, height, width=width)
plt.xticks(x_pos, bars, fontsize=11, fontweight='bold')
plt.title('Error vs. Latitude', fontweight='bold', fontsize=14)
#plt.xticks(range(18),['ST1\n(-85,-120)','ST2\n(-85,0)','ST3\n(-85,120)','ST4\n(-50,-120)','ST5\n(-50,0)','ST6\n(-50,120)','ST7\n(-20,-120)','ST8\n(-20,0)','ST9\n(-20,120)','ST10\n(20,-120)','ST11\n(20,0)','ST12\n(20,-120)','ST13\n(50,-120)','ST14\n(50,0)','ST15\n(50,120)','ST16\n(85,-120)','ST17\n(85,0)','ST18\n(85,120)'], fontweight='bold', fontsize=10)
plt.xlabel('Station Nº Lat.', fontweight='bold', fontsize=10)
plt.ylabel('Error', fontweight='bold', fontsize=10)
plt.legend()
plt.show()
