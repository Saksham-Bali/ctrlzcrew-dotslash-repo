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

