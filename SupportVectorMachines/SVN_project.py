#famous iris data set as an example of discriminatn analysis: measured length and width of petals and sepals :)
#Which flower species seems to be the most separable?

#import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 

#use seaborn to get the iris data set using iris = sns.load_dataset('iris')
iris = sns.load_dataset('iris')

#print(iris.head)
#create a pairplot of the data set. Which species seems to be the most separable?
#sns.pairplot(iris, hue='species')
#plt.show()
#create a KDE plot of sepal_length vs. sepal width for setosa species of flowers
#setosa = iris[iris['species']=='setosa']
#sns.kdeplot(setosa['sepal_width'], setosa['sepal_length'], cmap = 'plasma', shade=True, shade_lowest=False)
#don't shade the lowest values of the graph

#Split your data into a training set and a testing set
from sklearn.model_selection import train_test_split
X = iris.drop('species', axis=1)
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.80, random_state=42)
#Train the model: Call the SVC model from sklearn and fit the model to the training data.
from sklearn.svm import SVC
modelSVC = SVC()
modelSVC.fit(X_train, y_train)
#Now get predictions from the model and create a confusion matrix and classifcation report.
from sklearn.metrics import confusion_matrix, classification_report
predictions = modelSVC.predict(X_test)

#Import GridsearchCV from SciKit Learn
from sklearn.model_selection import GridSearchCV

#create a dictionary called param_grid and fill out some parameters for C and gamma
param_grid = {'C': [0.1, 1, 10, 100 ], 'gamma': [1, 0.1, 0.01, 0.001], 'degree': [1, 2, 3]}

#Create a GridSearchCV object and fit it to the training data
gridSearch = GridSearchCV(SVC(kernel='poly'), param_grid, verbose=2)
gridSearch.fit(X_train, y_train)

print('/n')
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print('/n')
print(gridSearch.best_params_)
print(gridSearch.best_estimator_)

#Now take that grid model and create some predictions using the test set and create
#classification reports and confusion matrices for them. Were you able to improve?
optimizedPredictions = gridSearch.predict(X_test)
print('/n')
print('Optimized predictions:')
print('/n')
print(confusion_matrix(y_test, optimizedPredictions))
print(classification_report(y_test, optimizedPredictions))