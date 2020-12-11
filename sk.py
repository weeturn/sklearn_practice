import pandas as pd
import numpy as np
from numpy import array
from pathlib import Path
from keras.utils import to_categorical
from sklearn import preprocessing 
from keras.layers import Dense
from keras import regularizers
from keras.models import Sequential
from keras.callbacks import EarlyStopping

def preprocess(choice):
    if choice == 1:
        data = pd.read_csv("train.csv", header=0, skip_blank_lines=True) #read train.csv
    else:
        data = pd.read_csv("test.csv", header=0, skip_blank_lines=True) #read test.csv
        
    # turn attributes into one-hot encoding
    month_encoded = pd.get_dummies(data['Attribute11'])
    data = data.drop('Attribute11', axis = 1)
    data = data.join(month_encoded)
    print("m:", data.shape)
    
    return_encoded = pd.get_dummies(data['Attribute16'])
    data = data.drop('Attribute16', axis = 1)
    data = data.join(return_encoded)
    print("r:", data.shape)
    
    #turn true false into 0 1
    data['Attribute17'] = data['Attribute17'].astype(np.int64)
    if choice == 1:
        data['Attribute18'] = data['Attribute18'].astype(np.int64)
    
    return data

# Split label out
data = preprocess(1)
y_train = data.pop('Attribute18')

x_train = data
print(x_train.shape)

classifier = Sequential()
#First Hidden Layer
classifier.add(Dense(28, activation='relu', kernel_initializer='random_normal', 
                     input_dim=28,kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.0001))) #28 512 128 1
#Second Hidden Layer
classifier.add(Dense(512, activation='relu', kernel_initializer='random_normal',
                     kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.0001)))
#Third Hidden Layer
classifier.add(Dense(128, activation='relu', kernel_initializer='random_normal',
                     kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.0001)))
#Output Layer
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal',
                     kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.0001)))

#Compiling the neural network
classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])

class_weight = {0: 1.,
                1: 9.}
#Earlystopping
EarlyStopping(monitor='val_loss', patience=5, verbose=2)
#Fitting the data to the training dataset
classifier.fit(x_train,y_train, batch_size=10, epochs=20, class_weight=class_weight)

#Predict
x_test = preprocess(2)
print(x_test.shape)
y_pred=classifier.predict(x_test)
y_pred =(y_pred>0.5) # 0.35
print(y_pred)

temp = 0
for x in y_pred:
    if x == True:
        temp = temp+1
print(1-temp/1444)

df = pd.DataFrame(columns=['id', 'ans'])
ids = [float(x) for x in range(1444)]
y_pred = y_pred.astype(np.int64)
df['id'] = ids
df['ans'] = y_pred

# Output to csv
df.to_csv('result1.csv', index=False)