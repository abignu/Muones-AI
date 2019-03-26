


#cargamos data
from _lectorHDF import *
import matplotlib.pyplot as plt 


# FNN
fnn = Sequential()
#input layer
fnn.add(Dense(8, activation='relu'))
#hidden layers
fnn.add(Dense(7, activation='relu'))
fnn.add(Dense(7, activation='relu'))
fnn.add(Dense(7, activation='relu'))
#output layer
fnn.add(Dense(1, activation='linear'))
#compilamos el modelo
fnn.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

print('entrenando...')
#entrenamos el modelo
history = fnn.fit(train_data_X, train_data_Y, validation_split = 0.2, epochs=50, batch_size=100)

#evaluamos el modelo
predicciones = fnn.predict(test_data_X)
contador = np.array([])

#guardamos el modelo y los weights
model_json = fnn.to_json()
with open('mezclado/mezclado.json', 'w') as json_file:
	json_file.write(model_json)

fnn.save_weights('mezclado/mezclado.h5')
print('Guardado los weights y el json')

np.savetxt("mezclado/predicciones.csv", predicciones, delimiter=",")

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
