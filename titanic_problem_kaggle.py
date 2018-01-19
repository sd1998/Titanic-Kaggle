#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 18:20:48 2018

@author: shashvatkedia
"""

import pandas as pd
import numpy as np

dataset_train = pd.read_csv('train.csv')
y = dataset_train.iloc[:,1].values
dataset_train = dataset_train.drop(['PassengerId','Survived','Name','Ticket','Cabin'],axis=1)
dataset_train = dataset_train.fillna(dataset_train.mean())
dataset_train['Embarked'] = dataset_train['Embarked'].fillna(dataset_train['Embarked'].mode()[0])
X_train = dataset_train.iloc[:,:].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoderSex = LabelEncoder()
X_train[:,1] = labelEncoderSex.fit_transform(X_train[:,1])
labelEncoderEmbarked = LabelEncoder()
X_train[:,6] = labelEncoderEmbarked.fit_transform(X_train[:,6])

from sklearn.model_selection import train_test_split
X_train1,X_test,y_train,y_test = train_test_split(X_train,y,random_state=0,test_size=0.2)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train1 = scaler.fit_transform(X_train1)
X_test = scaler.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout         #To avoid overfitting

classifier = Sequential()

classifier.add(Dense(output_dim=5,activation='relu',input_dim=7,init='uniform'))
classifier.add(Dropout(0.04))
classifier.add(BatchNormalization())

classifier.add(Dense(output_dim=4,activation='relu',init='uniform'))
classifier.add(Dropout(0.05))
classifier.add(BatchNormalization())

#classifier.add(Dense(output_dim=4,activation='relu',init='uniform'))
#classifier.add(Dropout(0.05))
#classifier.add(BatchNormalization())

classifier.add(Dense(output_dim=1,activation='sigmoid',init='uniform'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

classifier.fit(X_train1,y_train,batch_size=10,nb_epoch=100)

predict = classifier.predict(X_test)
predict = (predict > 0.5)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predict))
