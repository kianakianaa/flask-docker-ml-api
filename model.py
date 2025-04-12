import pandas as pd
import pickle
import statsmodels.api as sm

# Load the Excel data
df = pd.read_excel('training_data.xlsx')

# Define features and target
X = df[['w', 'x']]  # Input features: treatment and covariate
y = df['y']         # Output: outcome variable

# Train the model
model = sm.OLS(y, X).fit()

# Save the trained model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
