'exec(%matplotlib inline)'
import pandas as pd
import importlib
import numpy as np
import keras
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import requests
#CALEA BAZEI DE DATE
df = pd.read_csv(r"C:\Users\Alin\Desktop\LICENTA\luni.csv")

# AFISARE
df.info()
# print(df.head())

# SEPARARE VARIABILE
#
X = df.iloc[:, 0:16].values
y = df.iloc[:, 16].values
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# print(X)
# print(y)

# MODELUL 2
# definire model
seed = 7
np.random.seed(seed)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(X, y):
    model = Sequential()
    model.add(Dense(12, input_dim=16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
# compilare model
    model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
# fixare model
    model.fit(X, y, epochs=10, batch_size=10)
# evaluare model
    scores = model.evaluate(X[test], y[test], verbose=0)
    # print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    # cvscores.append(scores[1] * 100)
    # print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

# # evaluate the keras model
# # print('Acuratete: %.2f' % (accuracy*100))
    predictions = model.predict_classes(X)

for i in range(1):
    print(type(X))
rezultat = X.shape
i = rezultat[0]-1
print(predictions[i], y[i])
url = "http://192.168.1.83:5000/set_data_ai"
ob = {"rezultat": predictions[i]}
print(requests.post(url, data=ob).text)
print('%s => %d (Asteapta %d)' % (X[i].tolist(), predictions[i], y[i]))

# # # SALVARE PREDICTIE 
# # df['pred'] = predictions
# # df.to_csv('data1.csv')


# #   PRELUARE DATE
# import csv
# import json
# import requests
# response = requests.get("http://192.168.1.83:5000/data_ML")
# data=response.json()
# with open('luni.csv', 'w') as f:
#     thewriter = csv.writer(f)
#     thewriter.writerow(["cantitate_hrana","data_hranire","data_ora","data_tratament","greutate","id_stup","presiune_stup","rame_goale",
#     "rame_mancare","rame_puiet","substanta_tratament","temperatura_exterior","temperatura_stup","umiditate_exterior","umiditate_stup",
#     "vant","stare"])
#     for i in range(len(data)):
#        thewriter.writerow([
#         data[i]["cantitate_hrana"],
#         data[i]["data_hranire"],
#         data[i]["data_ora"],
#         data[i]["data_tratament"],
#         data[i]["greutate"],
#         data[i]["id_stup"],
#         data[i]["presiune_stup"],
#         data[i]["rame_goale"],
#         data[i]["rame_mancare"],
#         data[i]["rame_puiet"],
#         data[i]["substanta_tratament"],
#         data[i]["temperatura_exterior"],
#         data[i]["temperatura_stup"],
#         data[i]["umiditate_exterior"],
#         data[i]["umiditate_stup"],
#         data[i]["vant"],
#         data[i]["stare"]
#         ])

# # SALVARE DATE
#  df.to_csv('luni.csv')
