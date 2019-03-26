
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
#acomodamos data
import pandas as pd
import numpy as np

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
       'azimuth', 'raw_risetime', 'falltime', 'muon_total_signal']].copy()
comprobacion_real = comprobacion[['energy_mc', 'zenith_mc', 'r', 'total_signal', 'trace_length',
       'azimuth', 'raw_risetime', 'falltime', 'muon_total_signal']].copy()

df_real.energy_mc = np.log10(df_real.energy_mc) #paso la energia a base logaritmica
comprobacion_real.energy_mc = np.log10(comprobacion_real.energy_mc) #paso la energia a base logaritmica

#normalizo
df_real.energy_mc = (df_real.energy_mc - np.mean(df_real.energy_mc)) / np.std(df_real.energy_mc)
df_real.zenith_mc = (df_real.zenith_mc - np.mean(df_real.zenith_mc)) / np.std(df_real.zenith_mc)
df_real.r = (df_real.r - np.mean(df_real.r)) / np.std(df_real.r)
df_real.total_signal = (df_real.total_signal - np.mean(df_real.total_signal)) / np.std(df_real.total_signal)
df_real.trace_length = (df_real.trace_length - np.mean(df_real.trace_length)) / np.std(df_real.trace_length)
df_real.azimuth = (df_real.azimuth - np.mean(df_real.azimuth)) / np.std(df_real.azimuth)
df_real.raw_risetime = (df_real.raw_risetime - np.mean(df_real.raw_risetime)) / np.std(df_real.raw_risetime)
df_real.falltime = (df_real.falltime - np.mean(df_real.falltime)) / np.std(df_real.falltime)
df_real.muon_total_signal = (df_real.muon_total_signal - np.mean(df_real.muon_total_signal)) / np.std(df_real.muon_total_signal)

df_real = df_real.values #lo paso a array
comprobacion_real = comprobacion_real.values #lo paso a array

#armamos los targets
target = df[['Fuente']].copy()
comprobacion_target = comprobacion[['Fuente']].copy()
#target.muon_total_signal = (target.muon_total_signal - np.mean(target.muon_total_signal)) / np.std(target.muon_total_signal)

target = target.values #lo paso a array
comprobacion_target = comprobacion_target.values

target = np.ravel(target)

#codificamos las fuentes
le = preprocessing.LabelEncoder()
le.fit(target)
target = le.transform(target)



train_data_X, test_data_X, train_data_Y, test_data_Y = train_test_split(df_real, target, shuffle=True, test_size=0.2) #20% es test data
#SVM
svc_model = SVC()

print('entrenando...')


svc_model.fit(train_data_X, train_data_Y)

y_predict = svc_model.predict(test_data_X)

from sklearn.metrics import classification_report, confusion_matrix

#cm = np.array(confusion_matrix(test_data_Y, y_predict, labels=fuente))

print(classification_report(test_data_Y, y_predict))
