import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.utils import shuffle

#from _lectorHDF import *


#dataset ppal
df = pd.read_hdf('mezclado/df_mezclado.hdf')
comprobacion = pd.read_hdf('mezclado/comprobacion.hdf')

#agregamos tag de fuente
fuente = ['protones', 'helio', 'nitrogeno', 'hierro']

#ahora se lo agregamos a los dataframes
df['Fuente'] = fuente[0]
df['Fuente'].iloc[1:50000] = fuente[0]
df['Fuente'].iloc[50000:100000] = fuente[1]
df['Fuente'].iloc[100000:150000] = fuente[2]
df['Fuente'].iloc[150000:] = fuente[3]


comprobacion['Fuente'] = fuente[0]
comprobacion['Fuente'].iloc[1:100] = fuente[0]
comprobacion['Fuente'].iloc[100:200] = fuente[1]
comprobacion['Fuente'].iloc[200:300] = fuente[2]
comprobacion['Fuente'].iloc[300:] = fuente[3]


#df = shuffle(df) #shuffle
#armamos los inputs
print('armamos los inputs...')
df_real = pd.DataFrame()

df.zenith_mc = 1.0 / np.cos(df.zenith_mc) #le hacemos la secante al zenith
comprobacion.zenith_mc = 1.0 / np.cos(comprobacion.zenith_mc) #le hacemos la secante al zenith

df_real = df[['energy_mc', 'zenith_mc', 'r', 'total_signal', 'trace_length',
       'azimuth', 'raw_risetime', 'falltime']].copy()
comprobacion_real = comprobacion[['energy_mc', 'zenith_mc', 'r', 'total_signal', 'trace_length',
       'azimuth', 'raw_risetime', 'falltime']].copy()

df_real.energy_mc = np.log10(df_real.energy_mc) #paso la energia a base logaritmica
comprobacion_real.energy_mc = np.log10(comprobacion_real.energy_mc) #paso la energia a base logaritmica

#armamos los targets
target = df[['muon_total_signal']].copy()
comprobacion_target = comprobacion[['muon_total_signal']].copy()

sns.pairplot(comprobacion, hue='Fuente', vars=['energy_mc', 'zenith_mc', 'r', 'total_signal'], diag_kind='kde')#, 'trace_length', 'azimuth', 'raw_risetime', 'falltime'])
plt.show()

plt.figure(figsize=(20,12))
sns.heatmap(comprobacion.corr(), annot=True)
plt.show()



