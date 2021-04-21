import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from  sklearn.preprocessing import StandardScaler
data = pd.read_csv("data.csv")
# This takes all the rows, and all the columns exepct the last one, values collects them
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

ct  = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

le = LabelEncoder()
y = le.fit_transform(y)

#80% in training set 20% in test set
#this splits into x_train, x_test, y_train and y_test
x_train, x_test, y_train, y_test  = train_test_split(x, y, test_size=0.2,  random_state=1)



#normalization versus standardisation

#Xstand = (xi - mean(x)) ////////////////////


#standardization will always work

#Feautre scaling mus tbe applied after the split, on x_train and then x_test

#but then to scale the values of x_test you dont use the values of x_test x sub i, but rather
# you use the x values of training set to scale the test set.

sc = StandardScaler()

#the goal of standarisation is so the features stay in the same range
#you cant to do it to dumym varibles, and you wont be able to itnrepert the categorial varibles

#dont apply feature scalling to categorical varibles


x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])

print(x_test)

