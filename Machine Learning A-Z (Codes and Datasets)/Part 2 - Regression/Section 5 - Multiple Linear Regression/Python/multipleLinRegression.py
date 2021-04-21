import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from  sklearn.preprocessing import StandardScaler

# Importing the dataset
data = pd.read_csv('Data.csv')
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


ct  = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

x_train, x_test, y_train, y_test  = train_test_split(x, y, test_size=0.2,  random_state=1)

# Preprocessing of the data is now complete, we now need to do feature scaling

sc = StandardScaler()
x_train[:, :3] = sc.fit_transform(x_train[:, :3])
x_test[:, :3 ] = sc.transform(x_test[:, :3])

#Not enteraly sure about what to do with y_triana dn y_test, do we want to change those
#probably not, because we want that value to give an estimate.