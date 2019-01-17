# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

#Load Dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
df = pd.read_csv(url)
print(df)

#histograms
df.hist()
plt.show()

#Split-out validation dataset
X = df.iloc[:, 0:4]
y = df.iloc[:, 4]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

#Create a model
svm = LinearSVC()
svm.fit(X_train, y_train)
svm.predict(X_train)
print(svm.score(X_train, y_train))



