# -*- coding: utf-8 -*-
"""
Created on Mon May  4 22:03:46 2020

@author: Eric
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

originData = pd.read_csv("C0AI30.csv", encoding="gb18030", engine="python")
dataset = originData.iloc[:, [15]].values


sc = MinMaxScaler(feature_range= (0, 1))
trainSet = dataset[:-500]
trainSet = sc.fit_transform(trainSet)

xTrain = []
yTrain = []
for i in range(60, len(trainSet)):
    xTrain.append(trainSet[i-60:i])
    yTrain.append(trainSet[i, 0])
    
xTrain = np.array(xTrain)
yTrain = np.array(yTrain)

xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], xTrain.shape[2]))

model = Sequential()

model.add(
        LSTM(units=50, return_sequences=True, input_shape=(xTrain.shape[1], xTrain.shape[2])))
model.add(Dropout(0.1))

model.add(
        LSTM(units=50, return_sequences=True))
model.add(Dropout(0.1))

model.add(
        LSTM(units=50))
model.add(Dropout(0.1))

model.add(Dense(units=1))
model.compile(optimizer='adam', loss= 'mean_squared_error')

model.fit(xTrain, yTrain, epochs=100, batch_size=128)

#model.save("first")

xTest = []
yTest = []
testSet = dataset[-500:]
sc2 = MinMaxScaler(feature_range= (0, 1))
testSet = sc2.fit_transform(testSet)

for i in range(60, len(testSet)):
    xTest.append(testSet[i-60:i])
xTest = np.array(xTest)
xTest = np.reshape(xTest, (xTest.shape[0], xTest.shape[1], xTest.shape[2]))

predict = model.predict(xTest)
zeros = np.zeros(shape=(len(predict),4))
zeros[:,0] = predict[:,0]
predict = sc.inverse_transform(zeros)

    

plt.plot(dataset[-500:,0], color="red", label="Real Data")
plt.plot(predict[:,0], color="blue", label="Predicted Data")
plt.title("Temp predict")
plt.xlabel("Time")
plt.ylabel("Temp")
plt.legend()
plt.show()






