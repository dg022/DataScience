import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x, y)


percentErros = (abs(regressor.predict(x) - y) /y)*100
AveragePercentError  = sum(percentErros)/len(percentErros)
print("The accuarcy of this model is:", 100-AveragePercentError)

poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)

plt.scatter(x, y, color = 'red')
plt.plot(x, regressor.predict(x), color='blue')
plt.plot(x, lin_reg2.predict(x_poly), color='green')
plt.title('Experience Versus Salary')
plt.xlabel('Experience in Years')
plt.show()

percentErros = (abs(lin_reg2.predict(x_poly) - y) /y)*100
AveragePercentError  = sum(percentErros)/len(percentErros)
print("The accuarcy of this model is:", 100-AveragePercentError)