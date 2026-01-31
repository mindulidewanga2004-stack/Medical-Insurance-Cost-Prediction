import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import pickle

# LOAD YOUR ACTUAL CSV FILE (change filename to match yours)
df = pd.read_csv('insurance.csv')  # ← CHANGE THIS TO YOUR CSV FILENAME

# Clean smoker column to match your form (yes/no → 1/0)
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})

print(f"✅ Loaded YOUR dataset: {df.shape[0]} rows")
print("Sample data:")
print(df.head())

# Features and target (EXACT column order from your data)
X = df[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
y = df['charges']

# Preprocessing pipeline
numeric_features = ['age', 'bmi', 'children']
categorical_features = ['sex', 'smoker', 'region']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ])

# Full pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train on YOUR data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# Show accuracy on YOUR data
print(f"\n📊 Model Performance on YOUR data:")
print(f"Train R²: {pipeline.score(X_train, y_train):.4f}")
print(f"Test R²:  {pipeline.score(X_test, y_test):.4f}")

# Test with your first data point
test_pred = pipeline.predict(pd.DataFrame({
    'age': [19], 'bmi': [27.9], 'children': [0], 
    'sex': ['female'], 'smoker': [1], 'region': ['southwest']
}))[0]
print(f"\n🔍 Test: 19,Female,27.9,0,Yes,Southwest")
print(f"Actual:  $16,884.92")
print(f"Predicted: ${test_pred:,.2f}")

# Save model
with open('insurance_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("\n✅ Model trained on YOUR CSV data and saved!")
