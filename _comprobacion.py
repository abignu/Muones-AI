import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential  
from keras.layers import Dropout, Flatten, Dense 
import keras.backend as K
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error 
from xgboost import XGBRegressor

#abro lo csv
predicciones = pd.read_csv('mezclado/predicciones.csv')
test_data = pd.read_csv('mezclado/test_data_Y.csv')
x_test = pd.read_csv('mezclado/test_data_X.csv')

#pasamos los valores a arrays
predicciones = predicciones.values
test_data = test_data.values
x_test = x_test.values

#separamos las predicciones y las test_data encima de 25
rows = x_test.shape[0]
cols = x_test.shape[1]

#declaro vectores
y_encima25 = [] #np.ndarray(shape=(rows, 1))
preds_encima25 = []

#cogemos los que esten por encima de 5
for i in range(rows):
	if test_data[i] > 25:
		y_encima25.append(test_data[i])
		preds_encima25.append(predicciones[i])

#pasamos las listas a arrays
y_encima25 = np.asarray(y_encima25)
preds_encima25 = np.asarray(preds_encima25)

rows = x_test.shape[0]
cols = x_test.shape[1]

#declaro vectores
x_encima5 = np.ndarray(shape=(rows, cols))
y_encima5 = [] #np.ndarray(shape=(rows, 1))
preds_encima5 = []

#cogemos los que esten por encima de 5
for i in range(rows):
	for j in range(cols):
		if x_test[i,3] > 5:
			#x_encima5[i,j] = x_test[i,j]
			y_encima5.append(test_data[i])
			preds_encima5.append(predicciones[i])

#pasamos las listas a arrays
y_encima5 = np.asarray(y_encima5)
preds_encima5 = np.asarray(preds_encima5)


x_encima10 = np.ndarray(shape=(x_encima5.shape[0], x_encima5.shape[1]))
y_encima10 = [] #np.ndarray(shape=(x_encima5.shape[0], 1))
preds_encima10 = []
#ahora los que están encima de 10
for k in range(rows):
	for l in range(cols):
		if x_test[k,3] > 10:
			#x_encima10[k,l] = x_encima5[k,l]
			y_encima10.append(test_data[k])
			preds_encima10.append(predicciones[k])

#pasamos las listas a arrays
y_encima10 = np.asarray(y_encima10)
preds_encima10 = np.asarray(preds_encima10)

#los restamos
diferencia = predicciones - test_data
diferencia5 = preds_encima5 - y_encima5
diferencia10 = preds_encima10 - y_encima10
diferencia25 = preds_encima25 - y_encima25 #energia


#hacemos la media y la desviacion estándar del array diferencia
media = np.mean(diferencia)
media5 = np.mean(diferencia5)
media10 = np.mean(diferencia10)

print('media = {}, media arriba de 5 = {}, media arriba de 10 = {}'.format(media, media5, media10))

desv_estandar = np.std(diferencia)
desv_estandar5 = np.std(diferencia5)
desv_estandar10 = np.std(diferencia10)

print('de = {}, de arriba de 5 = {}, de arriba de 10 = {}'.format(desv_estandar, desv_estandar5, desv_estandar10))
print('RMS = {}'.format(np.sqrt(np.mean(np.square(diferencia)))))
print('RMS encima 25 = {}'.format(np.sqrt(np.mean(np.square(diferencia25)))))

#contador = range(0, diferencia.shape)
plt.scatter(test_data, diferencia)
plt.legend(['Diferencia entre las pred y el real (energía)'])
plt.xlabel('Valor real (energía´)')
plt.ylabel('diferencia (energía)')
plt.title('Diferencia vs test_data')
plt.show()

#acá cargamos el modelo y hacemos comprobación
from keras.models import model_from_json
datosX = pd.read_csv('mezclado/comprobacion_X.csv')
datosY = pd.read_csv('mezclado/comprobacion_Y.csv')

json_file = open('mezclado/mezclado.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("mezclado/mezclado.h5")
print("Loaded model from disk")

loaded_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
preds = loaded_model.predict(datosX)
contador = range(0,100)


f, axarr = plt.subplots(4, sharex=False)
f.suptitle('Energia')
axarr[0].plot(contador, preds[:100])
axarr[0].plot(contador, datosY[:100])
axarr[0].set_title('Pred Protones')
axarr[0].legend(['predicciones', 'test_data_Y'])

axarr[1].plot(contador, preds[100:200])
axarr[1].plot(contador, datosY[100:200])
axarr[1].set_title('Pred Helio')
axarr[1].legend(['predicciones', 'test_data_Y'])

axarr[2].plot(contador, preds[200:300])
axarr[2].plot(contador, datosY[200:300])
axarr[2].set_title('Pred Nitrogeno')
axarr[2].legend(['predicciones', 'test_data_Y'])

axarr[3].plot(range(0,99), preds[300:])
axarr[3].plot(range(0,99), datosY[300:])
axarr[3].set_title('Pred Iron')
axarr[3].legend(['predicciones', 'test_data_Y'])

plt.show()
