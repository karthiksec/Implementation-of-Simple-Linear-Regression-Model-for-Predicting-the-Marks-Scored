# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

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
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Karthik G.
RegisterNumber: 212223220043 
*/
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores (1).csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
## Dataset:
![309006202-11e77615-8f88-49b0-903f-2d479269b99d](https://github.com/karthiksec/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147473368/4800b822-3b30-4d75-9939-216c6beab4fe)

##Head Values:
![309006505-542dcb51-3cd6-49cb-a127-92bf922232dd](https://github.com/karthiksec/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147473368/72167189-c58c-4ad6-867f-83373013c3a8)

##Tail Values:
![309006832-41ef1520-0a28-4322-b900-b9571d053b8b](https://github.com/karthiksec/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147473368/fc666f2b-b9f3-41e5-8e54-a2e01352815d)

##X and Y Values:
![309007067-20303b68-d7ec-46e8-b922-f9b5cbc891f1](https://github.com/karthiksec/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147473368/b172a57a-4558-43cf-a371-e6914e977bb4)

##Prediction values X and Y:
![309007887-b4cc2779-f5a9-421e-bf6f-24089bc9b84f](https://github.com/karthiksec/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147473368/9c73267e-cdf6-4e51-8daf-499dfcd60be2)

##MSE, MAE and RMSE:
![309008434-ed6b78fb-1e30-44a5-b7ac-8530c1432dc7](https://github.com/karthiksec/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147473368/b08ba4c3-54c8-4e1c-a377-f77022b5db5e)

##Training Set:
![309008625-338e9031-40a1-407f-81fd-d89801e1f939](https://github.com/karthiksec/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147473368/b6f7ac9e-4edc-45f1-bbc3-fb6df1a7bef2)
##Testing set:
![309008905-74a4ba65-603c-4b3c-b10c-992a861a15f4](https://github.com/karthiksec/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147473368/0eb18307-2b87-48de-8e4a-b266df387dc8)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
