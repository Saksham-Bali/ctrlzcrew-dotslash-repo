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

#standardizing the data
standard_df = StandardScaler()
standard_df.fit(X_train)
X_train = standard_df.transform(X_train)
X_test = standard_df.transform(X_test)

#training the model usiung SVM, classifying the data based on hyperplane
model = svm.SVC(kernel = 'linear')
model.fit(X_train, Y_train)

#evaluating the model
#checking with the trained data without labels
X_train_pred = model.predict(X_train)
train_score = accuracy_score(Y_train, X_train_pred)
#accuracy score of test data
X_test_pred = model.predict(X_test)
pred_score = accuracy_score(Y_test, X_test_pred)

#creating a system for predictions
input_data = (162.56800,198.34600,77.63000,0.00502,0.00003,0.00280,0.00253,0.00841,0.01791,0.16800,0.00793,0.01057,0.01799,0.02380,0.01170,25.67800,0.427785,0.723797,-6.635729,0.209866,1.957961,0.135242)
input_arr = np.asarray(input_data)
#reshaping array
arr_reshaped = input_arr.reshape(1,-1)
#standardizing the input
std_arr = standard_df.transform(arr_reshaped)

#output
pred = model.predict(std_arr)
print(pred)

if pred[0] == 0:
    print('The person does not have Parkinsons Disease')
else:
    print('The person has Parkinsons Disease')
