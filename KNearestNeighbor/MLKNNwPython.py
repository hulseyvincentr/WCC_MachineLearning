#import the necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 


df = pd.read_csv('ClassifiedData.csv', index_col=0)
#print(df.head()) #<- check .csv was imported correctly, it was :)

#the scale of the variable matters, this will affect the way KNN classifies data.
#we need to standardize the scaling usking sklearn
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() #create instance of standard scaler (just like you would for normal ML algorithm)
scaler.fit(df.drop('TARGET CLASS', axis = 1)) #fit the scaler to your target class (all the feature columns)
#transform the data using the scaler
scaled_features = scaler.transform(df.drop('TARGET CLASS', axis = 1))
#print(scaled_features) #check values, looks good!
columnNames = df.columns[:-1] #make an array of the .csv's column names EXCEPT for the last column ('target class' column)
print(columnNames)
df_feat = pd.DataFrame(scaled_features, columns= columnNames)
#print(df_feat.head) #now data is standardized and can be put into a KNN algorithm

from sklearn.model_selection import train_test_split #sklearn.cross_validation seems to have been replaced as sklearn.model_selection
#source on rename: https://stackoverflow.com/questions/30667525/importerror-no-module-named-sklearn-cross-validation
X = df_feat
y = df['TARGET CLASS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=101)
#Now we ae our trained data :)

#use KNN to use elbow method to choose K value
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
pred = knn.predict(X_test) #this predicts the class of "TARGET CLASS". This will be binary, e.g. 0 or 1
#print(pred)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
#pretty good model w. k = 1, let's see if better k value will improve this using the elbow method
error_rate = [] #empty array
#iterate the model using different k values, and find the one with the lowest error rate
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors= i) #import the model at the given k value
    knn.fit(X_train, y_train) #fit the model to the training set
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i !=y_test)) #find the average number of times the prediction did not equal
    #the data, and append it to the error_rate array
plt.figure(figsize=(10,6))
plt.plot(range(1,40), error_rate, color = 'blue', linestyle ='dashed', marker = 'o', markerfacecolor = 'red', markersize = 10)
plt.title('Error rate vs K value')
plt.xlabel('K')
plt.ylabel('error rate')
plt.show()
#from the plot, we can see there is a minimum around k = 17. Choose k = 17 for our model
knn = KNeighborsClassifier(n_neighbors=17)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
print(confusion_matrix(y_test, pred))
print('/n') #print a new line between the values
print(classification_report(y_test, pred))
#accuracy is 95% now :D
