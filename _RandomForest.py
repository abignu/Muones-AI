from _lectorHDF import *

#aca aplico random forest
model = RandomForestRegressor()

#entrenamos
model.fit(train_data_X, train_data_Y)

#hacemos predicciones
y_predictions = model.predict(test_data_X)

score = model.score(test_data_X, test_data_Y)

print(score) #R^2 de la regresión con random forests R^2 ~ 0.96
