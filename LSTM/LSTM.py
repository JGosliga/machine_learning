import kagglehub
from zipfile import ZipFile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Download latest version
path = kagglehub.dataset_download("suyashlakhani/apple-stock-prices-20152020")
print("Path to dataset files:", path)

# Load data
df = pd.read_csv(f"{path}/AAPL.csv")
data = df['open'].values.reshape(-1, 1)  # Replace 'value_column' with your column name

print(len(data), "data points loaded.")

# Normalize data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Prepare sequences
def create_sequences(data, seq_length):
	X, y = [], []
	for i in range(len(data) - seq_length):
		X.append(data[i:i+seq_length])
		y.append(data[i+seq_length])
	return np.array(X), np.array(y)

seq_length = 50
X, y = create_sequences(scaled_data, seq_length)

# Split into train and test
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build LSTM model
model = Sequential([
	LSTM(50, return_sequences=False, input_shape=(seq_length, 1)),
	Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Train model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Predict
y_pred = model.predict(X_test)
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test)

# Visualize
plt.figure(figsize=(10,6))
plt.plot(y_test_inv, label='True')
plt.plot(y_pred_inv, label='Predicted')
plt.legend()
plt.title('LSTM Performance')
plt.show()