# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import and Load Data: Import required libraries, load the dataset, and drop unnecessary columns.
2. Preprocess Data: Convert categorical columns to numerical codes and separate features (x) and target (y).
3. Train Model: Initialize parameters, define sigmoid and loss functions, and apply gradient descent to optimize weights.
4. Predict and Evaluate: Make predictions, calculate accuracy, and display results for both the dataset and new students.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Jayakumar B 
RegisterNumber: 212223040073 
*/
```
```PY
# Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load and Prepare the Dataset
data = pd.read_csv('d:/chrome downloads/Placement_Data.csv')

# Drop Unnecessary Columns
data = data.drop(['sl_no', 'salary'], axis=1)

# Convert Categorical Columns to category Type
categorical_cols = ["gender", "ssc_b", "hsc_b", "degree_t", "workex", "specialisation", "status", "hsc_s"]
for col in categorical_cols:
    data[col] = data[col].astype('category')

# Convert Categories to Numerical Codes
for col in categorical_cols:
    data[col] = data[col].cat.codes

# Separate Features (X) and Target (y)
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Initialize Model Parameters
theta = np.random.randn(x.shape[1])
Y = y

# Define the Sigmoid Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define the Loss Function
def loss(theta, X, y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

# Implement Gradient Descent
def gradient_descent(theta, X, y, alpha, num_iterations):
    m = len(y)
    for _ in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h - y) / m
        theta -= alpha * gradient
    return theta

# Train the Logistic Regression Model
theta = gradient_descent(theta, x, y, alpha=0.01, num_iterations=1000)

# Define Prediction Function
def predict(theta, X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred

# Make Predictions and Compute Accuracy
y_pred = predict(theta, x)
accuracy = np.mean(y_pred.flatten() == y)
print("Accuracy:", accuracy)
print("Predicted Values:", y_pred)

# Predict placement for 2 new students
xnew = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])
y_prednew = predict(theta, xnew)
print("New Student 1 Prediction:", y_prednew)

xnew = np.array([[0, 0, 0, 0, 0, 2, 8, 2, 0, 0, 1, 0]])
y_prednew = predict(theta, xnew)
print("New Student 2 Prediction:", y_prednew)
```
## Output:

![image](https://github.com/user-attachments/assets/fcc280c1-881e-443e-964c-108820a9e68a)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

