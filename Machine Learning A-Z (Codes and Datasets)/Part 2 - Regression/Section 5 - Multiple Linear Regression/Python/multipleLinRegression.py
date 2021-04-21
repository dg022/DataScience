import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Importing the dataset
data = pd.read_csv('Data.csv')
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


ct  = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

x_train, x_test, y_train, y_test  = train_test_split(x, y, test_size=0.2,  random_state=1)