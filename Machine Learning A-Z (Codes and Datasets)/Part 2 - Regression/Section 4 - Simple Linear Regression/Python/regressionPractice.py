## first we want to import the data sets
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
data = pd.read_csv("Salary_Data.csv")


## Data pre-processing -----------------------------------------------------------------

#independat varible valeus
x = data.iloc[:, :-1].values

#depednat varible value/what we are trying predict
y = data.iloc[:, -1].values

x_train, x_test, y_train, y_test  = train_test_split(x, y, test_size=0.2,  random_state=1)

# ----------------------------------------------------------------------------------------

regressor = LinearRegression()
regressor.fit(x_train, y_train)


y_pred = regressor.predict(x_test)

plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Salary')
plt.show()

plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train,regressor.predict(x_train), color='blue')
plt.title('Salary vs Experience (Testing Set)')
plt.xlabel('Salary')
plt.show()



def predictSalary(exp, intercept, coeff):
    return exp*coeff + intercept

print(predictSalary(5.3,regressor.intercept_, regressor.coef_[0]  ))






#---------------------------------------------------------------
