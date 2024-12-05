# Train the model with multiple data points

# Import necessary libraries
# For numerical operations (not directly used here but often essential in ML workflows)
import numpy as np  
# For creating and manipulating datasets
import pandas as pd  
# For splitting the dataset into training and testing subsets
from sklearn.model_selection import train_test_split  
# For implementing a simple linear regression model
from sklearn.linear_model import LinearRegression  
# For evaluating the performance of the model
from sklearn.metrics import r2_score, mean_absolute_error  


# Step 1: The Dataset
# Create a small dataset using a Pandas DataFrame
data = pd.DataFrame({
    'MedInc': [10, 12, 15, 17, 20, 23, 25],
    'HouseAge': [20, 22, 25, 28, 30, 35, 40],
    'MedHouseVal': [4, 5, 6, 7, 8, 9, 10]
})

# Step 2: Splitting the Data
# Features (input variables for the model)
X = data[['MedInc', 'HouseAge']]
# Target variable (what we want to predict)
y = data['MedHouseVal']
# Split the data into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Training the Model
# Initialize the Linear Regression model
model = LinearRegression()
# Train the model on the training data
model.fit(X_train, y_train)

# Model coefficients and intercept
# Coefficients of the linear regression model
coefficients = model.coef_
# Intercept term of the linear regression model
intercept = model.intercept_

print("Step 3: Training the Model")
print(f"Coefficients: {coefficients}")
print(f"Intercept: {intercept}\n")

# Step 4: Testing the Model
# Use the trained model to predict values for the test data
y_pred = model.predict(X_test)
 # Compare actual vs. predicted values
results = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})

print("Step 4: Testing the Model")
print("Predicted vs Actual:")
# Display a table of actual vs predicted values
print(results, "\n")

# Step 5: Evaluating the Model
# R-squared (coefficient of determination)
r2 = r2_score(y_test, y_pred)
# Mean Absolute Error (average of absolute prediction errors)
mae = mean_absolute_error(y_test, y_pred)

print("Step 5: Evaluating the Model")
# Print R-squared to show model fit quality
print(f"R-squared: {r2}")
# Print Mean Absolute Error to show average prediction error
print(f"Mean Absolute Error: {mae}")




# Summary:
# Data Preparation: Dataset creation and feature-target splitting.
# Data Splitting: Divide data into training and testing sets.
# Model Training: Train the linear regression model using training data.
# Model Testing: Predict target values for the test dataset.
# Model Evaluation: Assess the model's performance with R-squared and Mean Absolute Error.





