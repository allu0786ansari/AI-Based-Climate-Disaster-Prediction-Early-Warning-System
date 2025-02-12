import tensorflow as tf
import numpy as np
import pandas as pd

def train_flood_model():
    data = pd.read_csv("../dataset/flood_data.csv")  # Load dataset
    X = data[['rainfall', 'temperature', 'humidity']]
    y = data['flood_risk']

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(X.shape[1], 1)),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=50, batch_size=16)
    model.save('../models/flood_model.h5')  # Save model
