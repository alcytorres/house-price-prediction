# Train the model with 1 data point 

# Step 1: Import necessary libraries
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 2: Define the dataset
# Features (MedInc, HouseAge) and Target (MedHouseVal)

# Features: MedInc, HouseAge
X = np.array([[10, 20], [15, 25], [20, 30]])  
# Target: MedHouseVal
y = np.array([4, 6, 8])                    

# Step 3: Split the dataset
# Training data: First two rows
X_train = X[:2]  # [[10, 20], [15, 25]]
y_train = y[:2]  # [4, 6]

# Test data: Last row
X_test = X[2:]   # [[20, 30]]
y_test = y[2:]   # [8]

# Step 4: Train the model
# Initialize and fit a simple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Get the coefficients and intercept from the trained model
coefficients = np.round(model.coef_, 1)   # Round coefficients
intercept = round(model.intercept_, 1)   # Round intercept
print("Model Coefficients:", coefficients)   # [0.4, 0.4]
print("Model Intercept:", intercept)         # 0.0

# Step 5: Test the model
# Use the model to predict house prices on the test data
y_pred = np.round(model.predict(X_test), 1)  # Round predictions
print("Test Input:", X_test)                 # [[20 30]]
print("Predicted Value:", y_pred[0])         # 8.0

# Step 6: Evaluate the model
# Compare predicted value with the actual value using RMSE
mse = mean_squared_error(y_test, y_pred)     # Mean Squared Error
rmse = round(np.sqrt(mse), 1)                # Root Mean Squared Error, rounded
print("Actual Value:", y_test[0])            # 8
print("Root Mean Squared Error (RMSE):", rmse) # 0.0 (perfect match)

# Summary: The model performs perfectly on this simple dataset

