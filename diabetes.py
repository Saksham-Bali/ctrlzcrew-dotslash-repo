# importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# loading the dataset
diabetes_dataset = pd.read_csv('diabetes.csv')
diabetes_dataset.head() #checking what data contains

# checking the number of rows and columns and all staistical measure of the data
diabetes_dataset.shape
diabetes_dataset.describe()

# Separaitng the data and labels
x = diabetes_dataset.drop(columns = 'Outcome',axis=1)
y = diabetes_dataset['Outcome']

# standardizing the data
scaler = StandardScaler()
scaler.fit(x)
standardized_data = scaler.transform(x)
x = standardized_data
y = diabetes_dataset['Outcome']
