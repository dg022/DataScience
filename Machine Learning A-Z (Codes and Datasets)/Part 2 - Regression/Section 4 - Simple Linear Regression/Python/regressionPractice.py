## first we want to import the data sets
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("Salary_Data.csv")


## Data pre-processing -----------------------------------------------------------------

#independat varible valeus
x = data.iloc[:, :-1].values

#depednat varible value/what we are trying predict
y = data.iloc[:, -1].values

x_train, x_test, y_train, y_test  = train_test_split(x, y, test_size=0.2,  random_state=1)

# ----------------------------------------------------------------------------------------



