#acomodamos data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

#dataset ppal
protones = pd.read_hdf('../protones/qgsjet-proton-ctodero.hdf')
helio = pd.read_hdf('../helium/qgsjet-helium-ctodero.hdf')
nitrogeno = pd.read_hdf('../nitrogeno/qgsjet-nitrogen-ctodero.hdf')
iron = pd.read_hdf('../iron/qgsjet-iron-ctodero.hdf')

protones = shuffle(protones) #shuffle
helio = shuffle(helio) #shuffle
nitrogneo = shuffle(nitrogeno) #shuffle
iron = shuffle(iron) #shuffle

protones = protones.values
helio = helio.values
nitrogeno = nitrogeno.values
iron = iron.values

rows = protones.shape[0]
cols = protones.shape[1]
df_mezclado = np.ndarray(shape=(200000, cols))

#creo dataset para hacer las predicciones por separado
comprobacion = np.ndarray(shape=(400, cols))
#los divido
print('mezclamos datasets 25 por ciento de cada uno...')

df_mezclado[0:50000,:] = protones[0:50000,:]
comprobacion[0:100,:] = protones[50000:50100,:]
df_mezclado[50000:100000,:] = helio[0:50000,:]
comprobacion[100:200,:] = helio[50000:50100,:]
df_mezclado[100000:150000,:] = nitrogeno[0:50000,:]
comprobacion[200:300,:] = nitrogeno[50000:50100,:]
df_mezclado[150000:200000,:] = iron[0:50000,:]
comprobacion[300:400,:] = iron[50000:50100,:]

df_mezclado = pd.DataFrame(data=df_mezclado, columns=['sim_id', 'energy_mc', 'zenith_mc', 'r', 'total_signal', 'trace_length', 'azimuth', 'raw_risetime', 'falltime', 'area_over_peak', 'muon_total_signal'])
comprobacion = pd.DataFrame(data=comprobacion, columns=['sim_id', 'energy_mc', 'zenith_mc', 'r', 'total_signal', 'trace_length', 'azimuth', 'raw_risetime', 'falltime', 'area_over_peak', 'muon_total_signal'])
'''
df_mezclado = protones[0:200000].copy()
df_mezclado.iloc[0:50000,:] = protones.iloc[0:50000,:]
df_mezclado.iloc[50000:100000,:] = helio.iloc[0:50000,:]
df_mezclado.iloc[100000:150000,:] = nitrogneo.iloc[0:50000,:]
df_mezclado.iloc[150000:200000,:] = iron.iloc[0:50000,:]

df_mezclado = df_mezclado.iloc[0:200000,:]
'''
#df_mezclado = shuffle(df_mezclado)

df_mezclado.to_hdf("../mezclado/df_mezclado.hdf", key='df', mode='w')
comprobacion.to_hdf("../mezclado/comprobacion.hdf", key='df', mode='w')

print('copiado con exito')