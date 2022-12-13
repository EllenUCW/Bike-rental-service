# Bike-rental-service
Bike rental service with machine learning application
Objectives of this project is to apply machine leanring models to predict bike counts and try to understand the infuential factors.
Data source: http://capitalbikeshare.com/system-data​

Step 1: import neccessary libraries

import pandas as pd # Used to load and read dataset
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt # Used to visual exploration
from sklearn.linear_model import LinearRegression # Used for Linear Regression model
from sklearn.ensemble import RandomForestRegressor # Used for Random Forest Regression model
from sklearn.svm import SVR # Used for SVM model
from sklearn.metrics import mean_absolute_error, r2_score # Used to evaluate regression models

Step 2: load dataset

import google.colab
uploaded = google.colab.files.upload()
df = pd.read_csv('hour.csv', index_col='instant')
df.head(2 )

Step 3: data preparation

Check types of data and non-null values with function info in Pandas
df.info()
Remove unwanted data: dteday, casual, registered
df.drop(columns = ['dteday' , 'casual', 'registered'], inplace=True ) 
Transfer numerical data into category
df['season']= df.season.astype('category')
df['yr']= df.yr.astype('category')
df['mnth']= df.mnth.astype('category')
df['hr']= df.hr.astype('category') 
df['holiday']= df.holiday.astype('category')
df['weekday']= df.weekday.astype('category')
df['workingday']= df.workingday.astype('category')
df['weathersit']= df.weathersit.astype('category')  
