import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.metrics import mean_absolute_error 

#abro lo csv
predicciones = pd.read_csv('mezclado/predicciones.csv')
test_data = pd.read_csv('mezclado/test_data_Y.csv')

print('MAE: ', mean_absolute_error(test_data, predicciones))

#los convierto a arrays
contador = []
contador = range(0,100)
predicciones = predicciones.iloc[:100].values
test_data = test_data.iloc[:100].values


#los ploteo
plt.plot(contador, predicciones)
plt.plot(contador, test_data)
plt.legend(['predicciones', 'test_data_Y'])
plt.xlabel('numero de datos')
plt.ylabel('energía')
plt.title('Comparación predicciones vs test_data')
plt.show()