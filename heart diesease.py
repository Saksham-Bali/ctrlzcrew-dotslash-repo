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
