# House Price Prediction Original Code

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load the California Housing dataset
housing = fetch_california_housing()

# Create a DataFrame
df = pd.DataFrame(housing.data, columns=housing.feature_names)

# Add the target variable
df['MedHouseVal'] = housing.target

# Display the first five rows
print("First five rows of the dataset:")
print(df.head())

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Features and target variable
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Scatter plot of Actual vs Predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Median House Value')
plt.ylabel('Predicted Median House Value')
plt.title('Actual vs Predicted Median House Value')
plt.show()

# Calculate residuals
residuals = y_test - y_pred

# Plot the distribution of residuals
plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=50, kde=True)
plt.xlabel('Residual')
plt.title('Distribution of Residuals')
plt.show()

# Function to predict house price based on user input
def predict_house_price():
    print("Enter the following details:")
    MedInc = float(input("Median Income: "))
    HouseAge = float(input("House Age: "))
    AveRooms = float(input("Average Rooms: "))
    AveBedrms = float(input("Average Bedrooms: "))
    Population = float(input("Population: "))
    AveOccup = float(input("Average Occupancy: "))
    Latitude = float(input("Latitude: "))
    Longitude = float(input("Longitude: "))
    
    # Create a NumPy array with the input values
    input_features = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])
    
    # Predict
    predicted_value = model.predict(input_features)
    print(f"\nPredicted Median House Value: {predicted_value[0]:.3f} (in 100,000 USD)")

# Call the function for user interaction
predict_house_price()

# Save the model
joblib.dump(model, 'house_price_model.pkl')
print("Model saved as 'house_price_model.pkl'")

# Load the model
loaded_model = joblib.load('house_price_model.pkl')
print("Model loaded successfully.")
