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

# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.2, stratify = y, random_state = 2)

# Training the model
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train,Y_train)

# checking the working of the data 
input_data = (4,110,92,0,0,37.6,0.191,30) #example data from the dataset
data_into_array = np.asarray(input_data)
reshaped_Data = data_into_array.reshape(1,-1)
std_data = scaler.transform(reshaped_Data)
output = classifier.predict(std_data)
print(output)
