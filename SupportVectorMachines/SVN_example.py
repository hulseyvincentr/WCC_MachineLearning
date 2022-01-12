import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

#use built-in breast cancer data from scikitlearn

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer() #cancer is an instance of calling the data

#print(cancer.keys()) #print out description of the data set
#the data set has: data, target, frame, target_names, DESCR, feature_names, filename
#print(cancer['DESCR']) #print out the description section of the data

df_feat = pd.DataFrame(cancer['data'], columns = cancer['feature_names'])
#make a data frame containing the data part of the cancer data set. the columns will be the feature_names
#from the cancer data set
#print(df_feat.head(2)) #see what the data frame looks like. Lots of numercal data
#cancer['target'] contains array of 0s and 1s. From cancer['target_names'] we see this means malignent or benign

#run the ML model here:
from sklearn.model_selection import train_test_split
train_test_split
X = df_feat #X is the data frame
y = cancer['target'] #y is the column of the data labeling it as malignant or benign

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.svm import SVC
model = SVC() #instantiate the model
model.fit(X_train, y_train) #fit the model to the data

#Notice that SVC has a LOT of different paramters, including degree and kernel. For now just use default values:
predictions = model.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, predictions))
print('\n')
print(classification_report(y_test, predictions))
#in the video, all ill-defined scoresa are set to 0, or benign, yielding a confusion matrix of: [0 66; 0 105]
#search for the best parameters using a grid search - scikitlearn has a built-in function to do this
#C controls the cost of mis-classification on training data. Large C gives low bias, high variance
#gamma is the free parameters of the radian basis function (rbf). Small gama is a gausian with a large variance,
#which leads to high bias and low variance.
from sklearn.model_selection import GridSearchCV
param_grid = {'C':[0.1, 1, 10, 100, 1000], 'gamma':[1, 0.1, 0.01, 0.001, 0.0001], 'degree': [1, 2, 3]} #make a dictionary where keys are the parameters that go into the model we're using (an SVC)
#test C and gamma values at each of the values in the input arrays

grid = GridSearchCV(SVC(), param_grid, verbose=3) #instantiate
#first, pass in estimator, the SVC. Then, pass in the param_grid
#higher values of verbose give greater text descriptions
grid.fit(X_train, y_train) #runs the same loop with cross validation to find the best combinations,
#then runs the model with the best parameter setting. You can grab the best parameter setting from the object:
print(grid.best_params_)
#in this case, the best params are C = 1, gamma = 0.0001
print(grid.best_estimator_)
#best estimator is C = 1, gamma = [-.00156 10; 3 102]
#find the confusion/classifcation report, you'll find accuracy []
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test, predictions))
print('\n')
print(classification_report(y_test, predictions))