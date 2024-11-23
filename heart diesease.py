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
