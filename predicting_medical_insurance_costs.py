# -*- coding: utf-8 -*-
"""Predicting_Medical_Insurance_Costs.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1VWNm6ewmXQWLHoIMJBtHBXFFlFmbrFWP

**Predicting_Medical_Insurance_Costs**

Import Necessary Packages
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

#Loading the dataset
medical_df=pd.read_csv('medical_insurance.csv')

medical_df

medical_df.head()

"""**About the Data:**

age:age of the insured person(numeric)

sex:gender of the insured person(categorical:male/female)

bmi:body mass index of the insured person(numeric)

children:number of children covered by the the insurance plan(numeric)

smoker:whether the person is a smoker or not(categorical:yes/no)

region:region where the insured person resides(categorical:northeast/northwest/southeast/southwest)

charges:Insurance charges/costs billed to the insured person(numeric)

Exploring Data & Analysing Data
"""

medical_df.shape

medical_df.info()

medical_df.describe()

"""Data Visualization"""

sns.displot(data=medical_df,x='age',height=3, aspect=3)
plt.title('Distribution of Age')
plt.show()

sns.displot(data=medical_df,x='sex',kind='hist',height=2, aspect=2)
plt.title('Gender Distribution')
plt.show()

medical_df['sex'].value_counts()

sns.displot(data=medical_df,x='bmi',height=3, aspect=3)
plt.title('BMI Distribution')
plt.show()

medical_df['bmi'].value_counts()

plt.figure(figsize=(4,4))
sns.countplot(data=medical_df,x='children')
plt.title('Count of Children')
plt.xlabel('Number of Children')
plt.ylabel('Count')
plt.show()

medical_df['children'].value_counts()

plt.figure(figsize=(4, 4))
sns.countplot(data=medical_df, x='smoker')
plt.title('Smokers vs Non-Smokers Distribution')
plt.show()

medical_df['smoker'].value_counts()

plt.figure(figsize=(4, 4))
sns.countplot(data=medical_df, x='region')
plt.title('Distribution of Regions')
plt.show()

medical_df['region'].value_counts()

"""Data Preprocessing: Convert categorical data to numerical"""

medical_df.replace({'sex':{'male':0,'female':1}},inplace=True)
medical_df.replace({'smoker':{'yes':0,'no':1}},inplace=True)
medical_df.replace({'region': {'southeast': 0, 'southwest': 1, 'northwest': 2, 'northeast': 3}}, inplace=True)

"""Feature Engineering"""

#Interaction between BMI and Smoker
medical_df['BMI_smoker'] = medical_df['bmi'] * medical_df['smoker']
medical_df['BMI_smoker']

#splitting data into features and target
X=medical_df.drop('charges',axis=1)
y=medical_df['charges']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=2)

X_train.shape

X_test.shape

"""Model Training: Linear Regression

"""

lg=LinearRegression()
lg.fit(X_train,y_train)
y_pred=lg.predict(X_test)

"""Model Evaluation for Linear Regression"""

print("Linear Regression R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

"""Model Evaluation: Random Forest

"""

rf = RandomForestRegressor(n_estimators=100, random_state=2)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

"""Model Evaluation for Random Forest"""

print("Random Forest R2 Score:", r2_score(y_test, y_pred_rf))
print("MAE:", mean_absolute_error(y_test, y_pred_rf))
print("MSE:", mean_squared_error(y_test, y_pred_rf))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))

"""Prediction System"""

#Interactive Prediction
print("Enter the following details for prediction:")
age = int(input("Age: "))
sex = int(input("Sex (0 for male, 1 for female): "))
bmi = float(input("BMI: "))
children = int(input("Number of children: "))
smoker = int(input("Smoker (0 for yes, 1 for no): "))
region = int(input("Region (0: southeast, 1: southwest, 2: northwest, 3: northeast): "))

#Input features
input_features = np.array([age, sex, bmi, children, smoker, region, bmi * smoker]).reshape(1, -1)
predicted_cost = rf.predict(input_features)
print(f"Predicted Medical Insurance Charges: ${predicted_cost[0]:.5f}")
