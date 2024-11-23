import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#checking 'target' distribution
heart_data['target'].value_counts()

#seperating 'target' and other features
x = heart_data.drop(columns='target',axis = 1)
y = heart_data['target']


#splitting data into training and testing data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,stratify = y,random_state = 2)

#model training
model = LogisticRegression()

#training logistic model with the training data
model.fit(x_train,y_train)

#training data accuracy
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction,y_train)


#test data accuracy
x_test_prediction = model.predict(x_test)
testing_data_accuracy = accuracy_score(x_test_prediction,y_test)

#data prediction
input_data = (37,1,2,130,250,0,1,187,0,3.5,0,0,2)
# input_data = (37,0,0,128,205,0,1,130,0,3.5,0,0,2)

np_array_form = np.asarray(input_data)
reshaped_array = np_array_form.reshape(1,-1)
prediction = model.predict(reshaped_array)
# print(prediction)
if(prediction[0] == 0):
    print("Person does not have heart disease")
else:
    print("Person has heart disease")
  
