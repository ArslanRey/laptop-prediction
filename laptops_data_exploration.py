#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Analysis/Exploration of Laptop Prices.
Source: https://www.kaggle.com/ionaskel/laptop-prices
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder



# load dataset
laptops = pd.read_csv("/Users/reyhanarslan/Spyder_Projects/Linear_Regression_Prices/laptop-prices/laptops.csv"
                      , encoding="latin-1")
laptops = laptops.drop("Unnamed: 0", axis='columns')
to_predict = "Price_euros"

# heatmap to visualize correlation between the features and also the price
laptop_enc = OrdinalEncoder()
laptop_enc.fit(laptops)
x_corr = laptop_enc.transform(laptops)

sns.heatmap(pd.DataFrame(x_corr, columns=laptops.columns).corr(), annot=True)

# see how many unique values are in each feature:
for l in laptops.columns:
    print(l, ':', len(laptops[l].unique()))
    
# show frequency of each unqiue value of a feature:
for col in laptops.columns:
    print(laptops[col].value_counts(sort=True), '\n')
    plt.figure(figsize=(15, 11))
    plt.hist(laptops[col], bins=len(laptops[col].unique()))
    plt.xticks(laptops[col].unique(), rotation=90, ha='center')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()



