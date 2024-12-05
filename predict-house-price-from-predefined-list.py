# Predict house price from predefined values

import joblib
import numpy as np

# Load the saved model
loaded_model = joblib.load('house_price_model.pkl')
print("Model loaded successfully.")

# Predict function with arguments
def predict_with_loaded_model(MedInc, HouseAge, AveRooms, AveBedrms, Population):
    # Create a NumPy array with the input values
    input_features = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population]])

    # Use the loaded model to predict
    predicted_value = loaded_model.predict(input_features)
    print(f"\nPredicted Median House Value: {predicted_value[0]:.3f} (in 100,000 USD)")

# Call the function with specific values
predict_with_loaded_model(10, 25, 6.0, 3, 1500)



# Here’s a realistic example input:
# MedInc: 10 (Equivalent to $100,000 median income)
# HouseAge: 25 (Average house age is 25 years)
# AveRooms: 6.0 (Average of 6 rooms per household)
# AveBedrms: 3.0 (Average of 3.0 bedrooms per household)
# Population: 1500 (Block has a population of 1,500 people)
