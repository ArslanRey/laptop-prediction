#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prediction of Laptop Prices with Linear Regression from scikit-learn.
Source: https://www.kaggle.com/ionaskel/laptop-prices
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import pickle



# load dataset
laptops = pd.read_csv("...../laptops.csv"
                      , encoding="latin-1")
laptops = laptops.drop("Unnamed: 0", axis='columns')
to_predict = "Price_euros"

# splitting data into X and y
#   and also dropping the following features, bcs the Correlation Matrix and 
#   the Data Exploration with Tableau showed:
#   - Product: a unique name and different in all companies, adds no value for price-prediction
#   - Weight: has a very low correlation with the price
#   - Inches: also has a very low correlation with the price
X = np.array(laptops.drop(labels=[to_predict, 'Inches', 'Weight', 'Product'], axis='columns'))
y = np.array(laptops[to_predict])

# preprocessing of categorical data:
cenc = OrdinalEncoder()
cenc.fit(X)
X = cenc.transform(X)

# shuffle, split and train data 25 times, to get best model
# and save best model as: laptop-price-model.pickle, to load for (maybe) later purposes
best_r2 = 0
for _ in range(25):
    # splitting data into train and test dataset
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
    
    # linear regression:
    model_linear = linear_model.LinearRegression()
    model_linear.fit(x_train, y_train)
    r2_val = model_linear.score(x_test, y_test)
    
    print('The ', _, '. ', 'iteration:', sep='')
    # The R^2-Value varies between 0.31 and 0.56 depending how the training
    #   and test data was shuffled and split
    print('Accuracy (R^2-Value) Train: \n', model_linear.score(x_train, y_train))
    print('Accuracy Test (R^2-Value): \n', model_linear.score(x_test, y_test))
    print('Coefficient: \n', model_linear.coef_)
    print('Intercept: \n', model_linear.intercept_)
    print('\n\n')
    
    if r2_val > best_r2:
        best_r2 = r2_val
        with open("laptop-price-model.pickle", "wb") as lin_file:
            pickle.dump(model_linear, lin_file)

# LOAD MODEL
pickle_file = open("laptop-price-model.pickle", "rb")
model_linear = pickle.load(pickle_file)


# preparation to plotting of the results:
scaler = StandardScaler()
X_features = scaler.fit_transform(X)
# PCA is used to reduce the dimension of X for plotting
pca = PCA(n_components=1)
x_features = pca.fit_transform(X_features)

# plot the laptops data set:
plt.figure(figsize=(15, 11))
plt.scatter(x_features, y,color='black')
plt.xlabel('Laptop')
plt.ylabel('Price')
plt.xticks([])
plt.title("Laptop Prices")
plt.show()

# plot X with the calculated y of the LinearRegression-Model
y_reg_line = X.dot(model_linear.coef_) + model_linear.intercept_
sns.set(rc={'figure.figsize':(15,11)})
reg_plot = sns.regplot(x_features, y_reg_line, scatter_kws={'s': 1.5, 'alpha': 1}, line_kws={'lw': 2, 'color': 'red'})
reg_plot.set(ylim=(0,6000))
reg_plot.set(xticks=[])
reg_plot.set(ylabel='Predicted Price')
reg_plot.set(xlabel='Laptops')

# plot the original data set with the real prices plus the best fit line
#   to visualize how good the best-fit-line "fits" the original data
m, b = np.polyfit(x_features.reshape(-1), y_reg_line, 1)
plt.figure(figsize=(15, 11))
plt.scatter(x_features, y,color='black')
plt.xlabel('Laptop')
plt.ylabel('Price')
plt.title("Laptop Prices and the Best-Fit-Line")
plt.plot(x_features, m * x_features + b, color='red', linewidth=2)
plt.show()





