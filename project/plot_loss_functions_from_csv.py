import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import sys


option = int(sys.argv[1])
station = sys.argv[2]
station2 = sys.argv[3]

if(option==1):
    csvFileName = 'history_log_'+station2+'.csv'
    fileSave = 'train_val_loss.png'
elif(option==2):
    csvFileName = 'history_log_'+station2+'_gru.csv'
    fileSave = 'train_val_loss_gru.png'
else:
    csvFileName = 'history_log_'+station2+'_cnn.csv'
    fileSave = 'train_val_loss_cnn.png'


df = pd.read_csv('../data/models/'+csvFileName)

#plt.rc('mathtext', fontset='stixsans')

fig, axs = plt.subplots(1, 1, figsize=(6, 3), tight_layout=False)
#plt.figure(figsize=(5, 3), dpi=200, constrained_layout=True)
#plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.4f}'))
#plt.title(station, fontsize=20, fontweight='bold')
axs.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
axs.yaxis.get_offset_text().set_fontsize(20)
axs.tick_params(axis='both', labelsize=20)
axs.set_xlim(0, 21)

dim = np.arange(1,25,1);
axs.xaxis.set_ticks([1,10,20])

domain = np.arange(1,len(df['loss'])+1,1)

axs.plot(domain, df['loss'], linewidth = '5')
axs.plot(domain,df['val_loss'], linewidth = '5')

fig.savefig('../data/models/'+fileSave)
plt.show()





