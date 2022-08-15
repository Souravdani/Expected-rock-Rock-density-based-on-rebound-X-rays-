# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 22:10:37 2022
Density based on X_ray different models 
@author: Soura
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_csv('G:\\Python\\ML and Data Science\\UNZIP_FOR_NOTEBOOKS_FINAL\\DATA\\rock_density_xray.csv')
df.columns=['signal','density']
sns.scatterplot(x='signal', y='density', data= df)

# We are not scaling our data (because simple data)
X= df['signal'].values.reshape(-1,1)
y= df['density']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)



#################### LINEAR REGRESSION MODEL ######################




from sklearn.linear_model import LinearRegression
lr_model= LinearRegression()
lr_model.fit(X_train, y_train)

lr_pred= lr_model.predict(X_test)
lr_pred ## Strange predicted values very close to each other....be alert something wrong with this model
from sklearn.metrics import mean_absolute_error, mean_squared_error
mse= mean_squared_error(y_test,lr_pred)
mse
mae= mean_absolute_error(y_test, lr_pred)
mae
rmse= np.sqrt(mse)
rmse
plt.figure(figsize=(12,8), dpi= 200)
sns.scatterplot(x='signal', y='density', data= df)

signal_range= np.arange(0,100)
signal_preds= lr_model.predict(signal_range.reshape(-1,1))
signal_preds

plt.figure(figsize=(12,8), dpi= 200)
sns.scatterplot(x='signal', y='density', data= df)
plt.plot(signal_range, signal_preds)
## You can cleary see that it is wrongly fitting the data
## We coudnt figure it out on basis of error metrics
## one way we could guess by lookig at predicted values




################# Helper function ##################
def run_model(model, X_train, y_train, X_test, y_test):
    
    # Fit the training model to the test data
    model.fit(X_train,y_train)
    
    # Get metrics
    predictions= model.predict(X_test)
    rmse= np.sqrt(mean_squared_error(y_test, predictions))
    mae= mean_absolute_error(y_test, predictions)
    print(f'rmse:{rmse}')
    print(f'rmse:{mae}')
    
    # plot result model signal range(0,100)
    
    signal_range= np.arange(0,100)
    output= model.predict(signal_range.reshape(-1,1))
    
    plt.figure(figsize=(12,8), dpi= 200)
    sns.scatterplot(x='signal', y='density', data= df, color='black')
    plt.plot(signal_range, output)



####################### POLYNOMIAL REGRESSION ####################

#model= LinearRegression()
#run_model(model, X_train, y_train, X_test, y_test)

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

pipe= make_pipeline(PolynomialFeatures(degree=6),LinearRegression()) # Ealier we did it in two steps 
# pipe will act like a normal model

run_model(pipe, X_train, y_train, X_test, y_test)


####################### KNN regression ######################
from sklearn.neighbors import KNeighborsRegressor

k_values= [1,5,10]
for n in k_values:
    model= KNeighborsRegressor(n_neighbors=n)
    run_model(model, X_train, y_train, X_test, y_test)


#################### Decision Tree ############
from sklearn.tree import DecisionTreeRegressor
model= DecisionTreeRegressor()
run_model(model, X_train, y_train, X_test, y_test)




################# Support Vector Regression ###########
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
svr= SVR()
param_grid={'C':[0.01,0.1,0.5,1,5,10,1000], 'gamma':['auto','scale']}
grid= GridSearchCV(svr, param_grid)

run_model(grid, X_train, y_train, X_test, y_test)



##################### RANDOM FOREST #####################

from sklearn.ensemble import RandomForestRegressor
rfr= RandomForestRegressor(n_estimators=10)
run_model(rfr, X_train, y_train, X_test, y_test)

############# BOOSTING ###########
from sklearn.ensemble import GradientBoostingRegressor,AdaBoostRegressor
model= GradientBoostingRegressor()
run_model(model, X_train, y_train, X_test, y_test)




















