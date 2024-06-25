#!/usr/bin/env python
# coding: utf-8

# In[137]:


import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
import tempfile

# Set Kaggle credentials as environment variables
os.environ['KAGGLE_USERNAME'] = 'tomerweissman661'
os.environ['KAGGLE_KEY'] = 'd3a379e67e038cd237bc5099338e6770'

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()

# Function to load a file from the Kaggle competition directly into a pandas DataFrame
def load_kaggle_competition_file(competition, file_name):
    with tempfile.TemporaryDirectory() as tmpdirname:
        file_path = os.path.join(tmpdirname, file_name)
        api.competition_download_file(competition, file_name, path=tmpdirname)
        return pd.read_csv(file_path)

# Check available files in the competition
files = api.competition_list_files('house-prices-advanced-regression-techniques').files
for f in files:
    print(f.name, f.size)

# Load train and test datasets
train_data = load_kaggle_competition_file('house-prices-advanced-regression-techniques', 'train.csv')
test_data = load_kaggle_competition_file('house-prices-advanced-regression-techniques', 'test.csv')


# In[203]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler


# Assuming train_data is already defined and contains 'LotFrontage', 'LotArea', and 'SalePrice' columns

# Handle missing values by filling with the mean value
front = train_data['LotFrontage'].fillna(train_data['LotFrontage'].mean())
area = train_data['LotArea'].fillna(train_data['LotArea'].mean())
year = train_data['YrSold'].fillna(train_data['YrSold'].mean())
mo = train_data['MoSold'].fillna(train_data['MoSold'].mean())

price = train_data['SalePrice'].fillna(train_data['SalePrice'].mean())

# Check for infinite values and replace them with the mean
front = np.where(np.isfinite(front), front, np.mean(front))
area = np.where(np.isfinite(area), area, np.mean(area))
mo = np.where(np.isfinite(mo), mo, np.mean(mo))
year = np.where(np.isfinite(year), year, np.mean(year))

price = np.where(np.isfinite(price), price, np.mean(price))

# Prepare input data X and output data Y
X = np.array([front, area, year, mo]).T
Y = np.array(price).reshape(-1, 1)

# Normalize the input features
# Scale the input features
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

M = X.shape[0]
test_size = int(round(M*0.15, 0))
X_test = X[:test_size]
Y_test = Y[:test_size]
X_train = X[test_size:]
Y_train = Y[test_size:]

# Define the model with L2 regularization
model = Sequential(
    [
        tf.keras.Input(shape=(X.shape[1],)),
        Dense(10, activation='relu', kernel_regularizer=l2(0.00001)),
        Dense(10, activation='relu', kernel_regularizer=l2(0.00001)),
        Dense(10, activation='relu', kernel_regularizer=l2(0.00001)),
        Dense(10, activation='relu', kernel_regularizer=l2(0.00001)),
        Dense(1, activation='linear', kernel_regularizer=l2(0.00001))
    ])

# Compile the model
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mean_squared_error',
    metrics=['mean_squared_error']
)

# Train the model
history = model.fit(
    X, Y,
    epochs=100,
    batch_size=32,
    validation_split=0.2
)

# Plot the training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()


# In[204]:


import numpy as np

def accuracy(X, Y, model):
    m = Y.shape[0]  # Number of samples
    total_squared_error = 0
    
    for i in range(m):
        y_true = Y[i]
        y_pred = model.predict(X[i].reshape(1, -1))

        # Calculate the squared error
        total_squared_error += (y_true - y_pred)**2
    
    # Calculate RMSE
    rmse = (total_squared_error / m)**0.5
    return rmse


# In[205]:


result = accuracy(X_test, Y_test, model)


# In[206]:


print(result[0])


# In[182]:


train_data.head()


# In[ ]:




