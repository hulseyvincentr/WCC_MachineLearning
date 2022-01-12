#Objective: Perform exploratory data analysis, standardize the data,
#do a train test split, and do KNN, choose a k value using the elbow
#method. Retrain using the best k value you found

# IMPORT pandas, seaborn, and the usual libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

#READ the KNN project data .csv file into a dataframe 
dataSet = pd.read_csv('KNN_Project_Data.csv')
#check you have the data by printing the first few rows
#print(dataSet.head())

#EXPLORATORY DATA ANALYSIS: 
#use seaborn on the dataframe to create a pairplot with the hue indicated by the TARGET CLASS column


#STANDARDIZE THE VARIABLES:
#import standard scaler from sklearn.preprocessing
from sklearn.preprocessing import StandardScaler

#create a StandardScaler() object called scaler
scaler = StandardScaler()
#Fit scaler to the features (drop the target class column)
#transform the features to a scaled version
#convert the scaled features to a dataframe (exclude the last column)
dataFeatures = dataSet.drop('TARGET CLASS', axis = 1)
dataLabel = dataSet['TARGET CLASS']
#print(dataFeatures.head)
fitScaledFeatures = scaler.fit(dataFeatures)
transformedFeatures = fitScaledFeatures.transform(dataFeatures)

#USE TRAIN_TEST_SPLIT TO SPLIT YOUR DATA INTO A TRAINING SET AND A TESTING SET
#import train_test_split from sklearn.MODEL SELECTION
from sklearn.model_selection import train_test_split
#create x values (the scaled features)
x_train, x_test, y_train, y_test = train_test_split(transformedFeatures, dataLabel, test_size = 0.25)
#create values (the target class column)
#print(y_train.shape) #<- use this to check dimensions 

#USE KNN
#import KNeightborsClassifier from scikit learn.neighbors
from sklearn.neighbors import KNeighborsClassifier
#Create a KNN model instance with n_neighbors = 1
knnModel = KNeighborsClassifier(n_neighbors=1)

#Fit this KNN model to the training data
knnModel.fit(x_train, y_train)

#PREDICTIONS AND EVALUATIONS
#Use the prediction method to predict values using your KNN model and X_test
predict = knnModel.predict(x_test)

#Create a confusion matrix and classification report
from sklearn.metrics import classification_report, confusion_matrix
#print(confusion_matrix(y_test, predict))
print(classification_report(y_test, predict))
creport = classification_report(y_test, predict, output_dict = True)
print(creport['accuracy'])

#CHOOSE A K VALUE
#Create a for loop that trains various KNN models with different k values, then keep track of the 
#error_rate for each of these models with a list.
errorRate = []

for k in range(1, 20):
    knnModel = KNeighborsClassifier(n_neighbors=k)
    knnModel.fit(x_train, y_train)
    predict = knnModel.predict(x_test)
    creport = classification_report(y_test, predict, output_dict = True)
    errorRate.append(creport['accuracy'])
print(errorRate)

#create the plot using the information from your for loop
sns.lineplot(y = errorRate, x = range(1, 20) )
plt.show()

# RETRAIN WITH NEW K VALUE
#Retrain your model with the best k value, re-do the classification report and the confusion matrix
kValue = 5

