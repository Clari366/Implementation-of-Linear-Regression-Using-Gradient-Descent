# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: clarissa k
RegisterNumber: 212224230047 
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate = 0.1, num_iters = 1000):
    X = np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors=(predictions - y ).reshape(-1,1)
        theta -= learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv("50_Startups.csv")
data.head()
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)
theta= linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
```

## Output:

<img width="558" height="222" alt="310191054-4015031e-f796-4985-b91e-2bcad67262af" src="https://github.com/user-attachments/assets/0e06b9c5-99d5-433a-9dfc-1e16579b75cf" />

VALUE OF X
<img width="225" height="713" alt="310191084-ab2f2037-d0d3-4828-ad78-8ca22db3770d" src="https://github.com/user-attachments/assets/26efb19c-91c1-4982-a279-814f34b29677" />

VALUE OF X1
<img width="343" height="707" alt="310191128-6a38641e-c102-473b-9bf2-a6afeaabcdc6" src="https://github.com/user-attachments/assets/da58f278-bba7-47cc-a436-4901eb0f6417" />

PREDICTED VALUE
<img width="247" height="46" alt="310191173-ae2ae2a4-16a5-4639-814d-5530c01c616d" src="https://github.com/user-attachments/assets/e9a6a462-82df-4c71-b19f-0270e4305710" />

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
