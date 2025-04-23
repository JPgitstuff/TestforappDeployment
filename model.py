# model.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import joblib

# Load and preprocess dataset
df = pd.read_csv("churn-bigml-80.csv")
df.drop(columns=['Area code', 'State'], inplace=True)

df['Total Calls'] = df['Total day calls'] + df['Total eve calls'] + df['Total night calls'] + df['Total intl calls']
df['Total Minutes'] = df['Total day minutes'] + df['Total eve minutes'] + df['Total night minutes'] + df['Total intl minutes']
df['Total Charge'] = df['Total day charge'] + df['Total eve charge'] + df['Total night charge'] + df['Total intl charge']
df['Plan'] = ((df['International plan'] == 'Yes') | (df['Voice mail plan'] == 'Yes')).astype(int)
df['Churn'] = df['Churn'].astype(int)

df.drop(columns=[
    'Voice mail plan', 'International plan', 'Number vmail messages',
    'Total eve calls', 'Total eve charge', 'Total eve minutes',
    'Total day calls', 'Total day charge', 'Total day minutes',
    'Total night calls', 'Total night charge', 'Total night minutes',
    'Total intl charge', 'Total intl minutes', 'Total intl calls'
], inplace=True)

X = df.drop("Churn", axis=1)
y = df["Churn"]

# Scale input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
joblib.dump(scaler, "scaler.pkl")

# Build the MLP model
from tensorflow.keras.models import save_model

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_scaled.shape[1],)),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train and save the model
model.fit(X_scaled, y, epochs=10, batch_size=32, verbose=1)
model.save("churn_prediction_model.h5")
