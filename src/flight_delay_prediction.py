# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import os

# Load dataset (replace 'flight_data.csv' with your dataset file)
data = pd.read_csv('flight_data.csv')

# Basic data exploration
print(data.head())
print(data.info())

# Feature selection and engineering
# Assuming dataset contains 'upstream_delay', 'departure_time', 'arrival_delay' columns
features = data[['upstream_delay', 'departure_time']]
labels = data['arrival_delay']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Support Vector Machine Regression Model
svr = SVR(kernel='rbf')
svr.fit(X_train_scaled, y_train)

# Predict on test set
y_pred = svr.predict(X_test_scaled)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Store results in a text file
if not os.path.exists('results'):
    os.makedirs('results')

with open('results/evaluation_metrics.txt', 'w') as f:
    f.write(f"Mean Squared Error: {mse}\n")
    f.write(f"R^2 Score: {r2}\n")

# Plotting the results
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.xlabel('Actual Arrival Delay')
plt.ylabel('Predicted Arrival Delay')
plt.title('Actual vs Predicted Arrival Delay')
plt.savefig('results/actual_vs_predicted.png')
plt.show()
