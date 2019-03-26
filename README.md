# Muones
En este repositorio de recogen una serie de códigos escritos en Python que analizan cuatro datasets diferentes de la creación de muones (partículas subatómicas) tras la colisión de partículas cósmicas con la alta atmósfera. Este trabajo se realizó para el departamento de Física Teórica y del Cosmos de la Universidad de Granada.

Encontramos dos códigos principales: 

- el primero se trata de una red neuronal que recibe los datos, procesados previamente en el archivo _lectorHDF.py, y los entrena para poder predecir a que dataset pertenece la simulación recibida. Es decir, decir a partir de qué se produjo la creación del muón. 

- el segundo es la aplicación del algoritmo SVM para la predicción cualitativa de las simulaciones en un dataset mezclado. Es decir, para poder distinguir las simulaciones mezcladas.
