# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 17:31:04 2015

OBJECTIVE: predict the total count of rented bikes (=demand)

Independent Variables
---------------------
datetime:   date and hour in "mm/dd/yyyy hh:mm" format
season:     Four categories-> 1 = spring, 2 = summer, 3 = fall, 4 = winter
holiday:    whether the day is a holiday or not (1/0)
workingday: whether the day is neither a weekend nor holiday (1/0)
weather:    Four Categories of weather
            1-> Clear, Few clouds, Partly cloudy, Partly cloudy
            2-> Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
            3-> Light Snow and Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
            4-> Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
temp:       hourly temperature in Celsius
atemp:      "feels like" temperature in Celsius
humidity:   relative humidity
windspeed:  wind speed

Dependent Variables
-------------------
registered: number of registered user
casual:     number of non-registered user
count:      number of total rentals (registered + casual)

@author: Nicolas
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn import metrics
import datetime as dt
import matplotlib.pyplot as plt


#---------------------- 1/ Data exploration -------------------------------
train_df = pd.read_csv('train.csv', header=0)

# feature engineering and preparing the training data set:
def create_new_features(df):
    df['year'] = df.datetime.map( lambda x: pd.to_datetime(x).year ).astype(int)
    df['month'] = df.datetime.map( lambda x: pd.to_datetime(x).month ).astype(int)
    df['dayofweek'] = df.datetime.map( lambda x: pd.to_datetime(x).dayofweek ).astype(int)
    df['hour'] = df.datetime.map( lambda x: pd.to_datetime(x).hour ).astype(int)

create_new_features(train_df)

# Create a box plot to observe each feature
train_df_boxplot = train_df.drop(['year','casual','registered','count'], axis = 1)  #then remove these columes from train_df
#train_df_boxplot.boxplot()

# Create a box plot to observe the demand by type of weather:
train_df.boxplot(column='count', by = 'weather')

# Create a box plot to observe the demand by hour:
train_df.boxplot(column='count', by = 'hour')

# Create a box plot to observe the demand by day of the week
train_df_boxplot.boxplot(column='count', by = 'dayofweek')
days = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
train_df_boxplot['namedayofweek'] = train_df['dayofweek'].map( days ) 

#Display the correlation of each column compared to the other ones:
print train_df[ ['weather','temp','atemp','humidity','windspeed','casual','registered','count'] ].corr()   
'''few inferences you can draw by looking at the above histograms:
-Variable temp is positively correlated with dependent variables (casual is more compare to registered) 
-Variable atemp is highly correlated with temp. 
-Windspeed has lower correlation as compared to temp and humidity'''


#---------------------- 2/ Prepare the test data set -------------------------------
test_df = pd.read_csv('test.csv', header=0)

create_new_features(test_df)

test_data_timestamp = test_df['datetime']                 # used for the output file creation

test_df = test_df.drop(['datetime'], axis = 1)

print "test_df = ", list(test_df)
print


#---------------------- 2/ Training Algo on Train Data-------------------------------

#OBJECTIVE: Predict the bike demand, registered and casual users separately

y = train_df['casual']                                                    # the response = only the column ("casual")
X = train_df.drop(['datetime','casual','registered','count'], axis = 1)   # the predictors = all columns except ("casual","registered","count") 
features_name = list(X)
print "y (train) = ", y.head(1), y.shape
print
print "X (train) = ", features_name, X.shape

# Convert back to a numpy array
y = y.values
X = X.values
X_test = test_df.values

forest = RandomForestClassifier(n_estimators=100)      
forest = forest.fit(X, y)                                                   #Fit the training data to "casual" 
print
print "---- Fit OK ----"
print

#---------------------- 3/ Predicting and saving -------------------------
# Then predict "casual" to the test data
y_pred = forest.predict(X_test).astype(int)

# could use the pandas to_csv function instead
predictions_file = open("bikeshare_prediction_RF.csv", "wb")    
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["datetime","casual"])
open_file_object.writerows(zip(test_data_timestamp, y_pred))
predictions_file.close()
print
print "---- Output OK ----"
print

# PRINT OUT VARIABLE IMPORTANCE
print pd.DataFrame(forest.feature_importances_, columns = ["Importance"], index = features_name).sort(['Importance'], ascending = False)
print


#------------------------- 5/ Confusion matrix -----------------------------------
# Show confusion matrix in a separate window
def show_confusion_matrix(yt, yp):
    cm = metrics.confusion_matrix(yt, yp)  # Compute confusion matrix
    plt.matshow(cm)  #generate a heatmap of the matrix
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
show_confusion_matrix(y_test, y_pred)



