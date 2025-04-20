import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Load example energy demand dataset (replace with your dataset)
# Expected format: a column 'demand' with time-series data
data = pd.read_csv("energy_demand.csv")
values = data["demand"].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(values)

# Prepare the data for LSTM (look back 10 time steps)
def create_dataset(data, look_back=10):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

look_back = 10
X, y = create_dataset(scaled_values, look_back)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split the data
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build the LSTM model
model = Sequential([
    LSTM(50, input_shape=(look_back, 1), return_sequences=True),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse")

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=1)

# Make predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test_inv = scaler.inverse_transform([y_test])

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(y_test_inv.T, label="Actual Demand")
plt.plot(predictions, label="Predicted Demand")
plt.title("Energy Demand Forecasting with LSTM")
plt.xlabel("Time Step")
plt.ylabel("Demand")
plt.legend()
plt.savefig("lstm_forecasting_result.png")
plt.close()

# Save the model
model.save("lstm_demand_forecasting.h5")
print("Model saved as lstm_demand_forecasting.h5")
