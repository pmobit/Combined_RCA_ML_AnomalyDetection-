#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 12:22:53 2022

@author: pooyan
"""

import pandas as pd
import numpy as np 
#### 
from sklearn.model_selection import train_test_split , StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
#df = pd.read_csv('./2K_1420_part_1234567features.csv',delimiter=';',index_col=False) 




df = pd.read_csv('/Users/pooyan/Documents/Emerson/data/data.csv', index_col=False, sep=";", header=0)
#set 0 as normal as 1 as abnormal
df.insert(3, 'target', 0)

# df.drop('Rows Header(1)')
df


   
index1 = []
index2 = []
index3 = [] 
df['target']= 0

#condition 1

index1 = df[df['RU.LNG.214TRA220.PV'] > df['RU.LNG.214TRA220.PV'].quantile(0.99)].index  |\
df[df['RU.LNG.214PRA218C.PV'] < df['RU.LNG.214PRA218C.PV'].quantile(0.01)].index |\
df[df['RU.LNG.214TRA476A.PV'] < df['RU.LNG.214TRA476A.PV'].quantile(0.01)].index  |\
df[df['RU.LNG.214PZA451A.PV'] < df['RU.LNG.214PZA451A.PV'].quantile(0.01)].index |\
df[df['RU.LNG.214PDRA425.PV'] < df['RU.LNG.214PDRA425.PV'].quantile(0.01)].index |\
df[df['RU.LNG.214PRA431.PV'] > df['RU.LNG.214PRA431.PV'].quantile(0.99)].index  |\
df[df['RU.LNG.214PRA434.PV'] > df['RU.LNG.214PRA434.PV'].quantile(0.99)].index  |\
df[df['RU.LNG.214PRA218.PV'] < df['RU.LNG.214PRA218.PV'].quantile(0.01)].index  |\
df[df['RU.LNG.214PDRA436.PV'] < df['RU.LNG.214PDRA436.PV'].quantile(0.01)].index |\
df[df['RU.LNG.214SZA458.PV'] < df['RU.LNG.214SZA458.PV'].quantile(0.01)].index |\
df[df['RU.LNG.214PRA221A.PV'] < df['RU.LNG.214PRA221A.PV'].quantile(0.01)].index |\
df[df['RU.LNG.214PRA221B.PV'] < df['RU.LNG.214PRA221B.PV'].quantile(0.01)].index |\
df[df['RU.LNG.214TR210.PV'] > df['RU.LNG.214TR210.PV'].quantile(0.99)].index |\
df[df['RU.LNG.214PRA218A.PV'] < df['RU.LNG.214PRA218A.PV'].quantile(0.01)].index
    
    
#condition 2    
    
#lube oil level    
index2 = df[df['RU.LNG.214LRSA450.PV'] <= 55].index | \
df[df['RU.LNG.214LRSA450.PV'] <= 45].index 

#condition 3
#lube oil pressure 
index3 = df[df['RU.LNG.214PZA451A.PV'] <= 1.77].index & \
df[df['RU.LNG.214PZA451A.PV'] <= 1.77].index 
  
#defining labels
df.loc[index1, 'target'] = 1  
df.loc[index2, 'target'] = 2  
df.loc[index3, 'target'] = 3      
 
print("Abnormal samples: ",len(df[df['target']==0]), "class2 samples: ",len(df[df['target']==1]) )    
print("Abnormal samples: ",len(df[df['target']==2]), "class2 samples: ",len(df[df['target']==3]) )   


X = df.loc[:,'RU.LNG.214PZA451A.PV':].to_numpy() # all measurement columns
y = df['target'].to_numpy() # labels 



#normalize
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X)
X_norm =scaler.transform(X)


#standard scaler
from sklearn.preprocessing import StandardScaler

x_standard = StandardScaler().fit_transform(X)



#2d pca

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x_standard)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])


X= principalComponents 


#train_X, test_X, train_y, test_y = train_test_split(X_norm, y, test_size=0.2, random_state=42)

#train_X, test_X, train_y, test_y = train_test_split(X_norm, y, test_size=0.2, random_state=42)
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


#build a model
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.losses import SparseCategoricalCrossentropy


# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam', metrics=["accuracy"])

#src: https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

model.summary()


history = model.fit(train_X, train_y, epochs=30, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=True)




# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
#model.add(Dropout(0.1)

model.add(Dense(4, activation = "softmax"))
model.add(Dense(1))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

model.summary()

yhat = model.predict(test_X)


scores = model.evaluate(test_X, yhat)

Dense_accuracy = scores[1]*100

print('Test accuracy: ', scores[1]*100, '%')



