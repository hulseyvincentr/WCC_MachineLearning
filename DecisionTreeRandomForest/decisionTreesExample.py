import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

dataFrame = pd.read_csv('DecisionTreeRandomForest/kyphosis.csv')
print(dataFrame.head())
#data lists patients who either had kyphosis or not after a corrective operation, their age,
# the number of vertebrae operated on, and the topmost vertebrae operated on in the procedure

#print(dataFrame.info())

#check the data out
#sns.pairplot(dataFrame, hue='Kyphosis')
#plt.show() 

#predict if patient has condition present after procedure
from sklearn.model_selection import train_test_split 
X = dataFrame.drop('Kyphosis', axis = 1) 
y = dataFrame['Kyphosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#train a single decision tree:
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier() #instantiate the renamed function
#print(X_train.head())
#print(y_train.head())
dtree.fit(X_train, y_train) #fit this model to the data
predictions = dtree.predict(X_test)#evaluate how well the decision tree predicts kyphosis presence
from sklearn.metrics import classification_report, confusion_matrix #import functions from sklearn
#print(confusion_matrix(y_test, predictions))
#print('\n') #add a new line for readability
#print(classification_report(y_test, predictions))

#compare these results from a random forest model
from sklearn.ensemble import RandomForestClassifier
rfcModel = RandomForestClassifier(n_estimators=200)
rfcModel.fit(X_train, y_train)
rfcPredictions = rfcModel.predict(X_test)
print(confusion_matrix(y_test, rfcPredictions))
print('\n')
print(classification_report(y_test, rfcPredictions))
#the random forest does a little better than our first tree. 
#random forests will outshine decision trees as data sets get larger.

print(dataFrame['Kyphosis'].value_counts())
#^out input data is imbalanced, there are more cases input that are absent
#than those that have kyphosis

