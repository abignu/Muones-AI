#acomodamos data
import pandas as pd
import numpy as np

print('a cargar keras...')

from keras.models import Sequential  
from keras.layers import Dropout, Flatten, Dense 
import keras.backend as K
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error 
from xgboost import XGBRegressor

print('a cargar datos...')
#dataset ppal
df = pd.read_hdf('mezclado/df_mezclado.hdf')
comprobacion = pd.read_hdf('mezclado/comprobacion.hdf')
df = shuffle(df) #shuffle
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



df_real = df_real.iloc[:150000,:].values #lo paso a array
comprobacion_real = comprobacion_real.values #lo paso a array


#armamos los targets
target = df[['muon_total_signal']].copy()
comprobacion_target = comprobacion[['muon_total_signal']].copy()

target = target.iloc[:150000,:].values #lo paso a array
comprobacion_target = comprobacion_target.values



from sklearn.model_selection import train_test_split


train_data_X, test_data_X, train_data_Y, test_data_Y = train_test_split(df_real, target, shuffle=True, test_size=0.2) #20% es test data


np.savetxt("mezclado/test_data_Y.csv", test_data_Y, delimiter=",")
np.savetxt("mezclado/test_data_X.csv", test_data_X, delimiter=",")

np.savetxt("mezclado/comprobacion_Y.csv", comprobacion_target, delimiter=",")
np.savetxt("mezclado/comprobacion_X.csv", comprobacion_real, delimiter=",")
