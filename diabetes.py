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
print("No of rows and columns:",diabetes_dataset.shape)
print("Printing the statistical measures of the dataset: ")
print(diabetes_dataset.describe())
