import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from  sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Importing the dataset
data = pd.read_csv('50_Startups.csv')
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


ct  = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

x_train, x_test, y_train, y_test  = train_test_split(x, y, test_size=0.2,  random_state=1)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

percentErros = (abs(regressor.predict(x_test) - y_test) /y_test)*100
AveragePercentError  = sum(percentErros)/len(percentErros)
y_pred = regressor.predict(x_test)

np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
print("the accruacy of the model is:", 100 - AveragePercentError)

#print( (abs(regressor.predict(x_test) - y_test) /y_test)*100)




 #in multiple linear regression there is no need to apply feature scaling, o.k