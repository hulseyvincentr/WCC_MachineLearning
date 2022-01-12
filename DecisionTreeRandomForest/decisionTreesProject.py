#Notes on assigment:
#Predict the not.fully.paid data column
#Make histograms based off of the data
#do some data to deal with categorical data

#import the typical libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#use pandas to read load_data.csv as a dataframe called loans
loans = pd.read_csv('loan_data.csv') #note the .csv files need to be outside of the 
#section folder and in the workspace for this to run

#Check out info(), head(), and describe() methods on the loans data frame
#print(loans.info())
#print(loans.head())
#print(loans.describe())

#exploratory data analysis
#create a histogram of the FICO distributions on top of each other
#loans[loans['credit.policy'] == 1]['fico'].hist(bins =35, color = 'blue', label = 'Credit Policy = 1', alpha = 0.6) 
#loans[loans['credit.policy'] == 0]['fico'].hist(bins = 35, color = 'red', label = 'Credit Policy = 0', alpha = 0.6)
#plt.legend()
#plt.xlabel('FICO score')
#plt.figure(figsize=(10,6))
#plt.show()
#more people have a credit policy equal to 1 instead of to 0. 

#create a histogram of the payment
#loans[loans['not.fully.paid'] == 0]['fico'].hist(bins = 35, color = 'green', label = 'Fully paid', alpha = 0.6)
#loans[loans['not.fully.paid'] == 1]['fico'].hist(bins =35, color = 'blue', label = 'Not fully paid', alpha = 0.6) 
#plt.legend()
#plt.xlabel('FICO score')
#plt.figure(figsize=(10,6))
#plt.show()

#make a plot showing the reason for the loan, with color indicating paid or unpaid
#sns.countplot(x='purpose', hue='not.fully.paid', data=loans, palette = 'Set1')
#plt.show()
#debt consolidation is the most popular reason for a loan

#show relation between fico score and interest rate for loans
#sns.jointplot(x='fico', y='int.rate', data=loans, color='purple')
#plt.show()

#show data with line of best fit, divide into columns based on if it wsa fully paid
#plt.figure(figsize=(11,7))
#sns.lmplot(y='int.rate', x='fico', data = loans, hue='credit.policy', col='not.fully.paid', palette='Set1')
#plt.show()

#DEAL WITH CATEGORICAL FEATURES
#we need to transform data using dummy variables s osklearn can understand the data. Use pd.get_dummies
#create a list of 1 element containing the string 'purpose' named cat_feats
cat_feats = ['purpose']

#use pd.get_dummies(loans, columns=cat_feats, drop_first = True) to create fixed parger dataframe
#that has new feature columns with dummy variables. Set this data frame as final_data
#drop_first prevents collinearity issues
final_data = pd.get_dummies(loans, columns=cat_feats, drop_first=True)
#print(final_data.head())

from sklearn.model_selection import train_test_split
X= final_data.drop('not.fully.paid', axis=1) #axis = 1 because it's a column
y= final_data['not.fully.paid'] #what we're trying to predict
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=101)

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier
dtree.fit(X_train, y_train) #fit model on training data

predictions = dtree.predict(X_test)

#from sklearn.metrics import classification_report, confusion_matrix

#print(classification_report(y_test, predictions))
#print(confusion_matrix,(y_test, predictions))
