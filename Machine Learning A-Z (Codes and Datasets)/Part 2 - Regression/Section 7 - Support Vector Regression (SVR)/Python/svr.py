import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from  sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

sc = StandardScaler()
sc_y = StandardScaler()
#the goal of standarisation is so the features stay in the same range
#you cant to do it to dumym varibles, and you wont be able to itnrepert the categorial varibles

#dont apply feature scalling to categorical varibles

#feature scaling for the levels
x = sc.fit_transform(x)
y = y.reshape(len(y),1)
y = sc_y.fit_transform(y)

regressor  = SVR(kernel='rbf')

regressor.fit(x, y)
r = sc_y.inverse_transform(regressor.predict(sc.transform([[6.5]])))
print(r)

plt.scatter(sc.inverse_transform(x), sc_y.inverse_transform(y), color = 'red')
plt.plot(sc.inverse_transform(x), sc_y.inverse_transform(regressor.predict(x)), color='blue')

plt.title('Experience Versus Salary')
plt.xlabel('Experience in Years')
plt.show()