#importing libraries
#to make arrays from dataset and work with them
import numpy as np
#to create dataframes from arrays and analyse it
import pandas as pd
#to split data into training and testing data
from sklearn.model_selection import train_test_split
#for standardizing the data
from sklearn.preprocessing import StandardScaler
#to implement the SVM(Support Vector Machine) model
from sklearn import svm
#to see the accuracy of the models
from sklearn.metrics import accuracy_score

#importing the data from the dataset
park_data = pd.read_csv('parkinsons.csv')
park_data.head(5)

#information on df
s = park_data.shape
park_data.info()

#finding null/missing values
park_data.isnull().sum()

#statistical info of the dataset
park_data.describe()

#distribution of target variable (status - 1(parkinson postive) or 0 (parkinson negative))
park_data['status'].value_counts()

#mean of df formed by grouping by status
park_data = park_data.drop(columns = ['name'], axis = 1)
park_data.groupby('status').mean()

#sepating the features and target
X = park_data.drop(columns=['status'], axis=1)
Y = park_data['status']

#splitting the data into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)