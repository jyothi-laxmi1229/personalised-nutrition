import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load your dataset
data = pd.read_csv('Indian_Food_Nutrition_Processing.csv')

# Select features (remove Username if non-numeric or encode otherwise; here we omit username)
X = data[['Age', 'Weight', 'Height']]
y = data['Calories_Needed']

# Train linear regression model
model = LinearRegression()
model.fit(X, y)

# Save the model to a file
joblib.dump(model, 'calorie_model.pkl')

print("✅ Model trained and saved.")
