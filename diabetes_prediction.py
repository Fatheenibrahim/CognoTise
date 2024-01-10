# -*- coding: utf-8 -*-
"""Diabetes Prediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1nVddrvRNAYMft06732shgHrxN-zIm8Wk

Diabetes Prediction
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import plotly.express as px

# to visualize missing datas in dataset
import missingno as msno

data = pd.read_csv('/content/diabetes_prediction_dataset.csv')

print(data.head())

print(data.columns)

print(data.describe())

print(data.describe().T)

print(data.info())

print(data.tail())

print(data.isnull())

# count of null values for each column\
print(data.isnull().sum())

# replacing 0 as null values
data_copy = data.copy(deep = True)

columns_to_replace = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history','bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes']

data_copy[columns_to_replace] = data_copy[columns_to_replace].replace(0, np.NaN)

print(data_copy.isnull().sum())

# histrogram visualization
print(data_copy.hist(figsize = (10,10))) # data.hist() for full size

# fillna -> to fill n or nan values

columns_to_fill = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history','bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes']

data_copy[columns_to_fill] = data_copy[columns_to_fill].fillna(data_copy[columns_to_fill].mean()) # data_copy[columns_to_fill].fillna(data_copy[columns_to_fill].mean()) directly to print the output with graph also

print(data_copy.hist(figsize = (10,10),color = 'green'))

# to print how many missing values are there
msno.bar(data_copy) #shows nothing

data

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

a = le.fit_transform(data.gender)
a

data.gender = a
data

b = le.fit_transform(data.smoking_history)
data.smoking_history = b

data.dtypes

x = data.drop("diabetes",axis = 1)

y = data.drop(['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history','bmi', 'HbA1c_level', 'blood_glucose_level'],axis=1)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=11,test_size=0.2)

x_train

y_train

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()

model.fit(x_train,y_train)

model.score(x_test,y_test)

y_predicted = model.predict(x_test)

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,y_predicted))

cf_matrix = confusion_matrix(y_test,y_predicted)

import seaborn as sns
sns.heatmap(cf_matrix, annot=True,fmt = 'd')