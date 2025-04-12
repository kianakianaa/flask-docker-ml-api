import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Load the Excel data
df = pd.read_excel('training_data.xlsx')

# Define features and target
X = df[['w', 'x']]  # Input features: treatment and covariate
y = df['y']         # Output: outcome variable

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the trained model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
