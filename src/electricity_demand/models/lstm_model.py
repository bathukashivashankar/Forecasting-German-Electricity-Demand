from typing import Tuple
"""
Part 6 LSTM training was executed in Google Colab due to local TensorFlow/Python 3.12 compatibility issues.
This script is kept as the reference implementation for the project.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras


def build_univariate_lstm(input_shape: Tuple[int, int]) -> keras.Model:
    """
    Build a simple univariate LSTM model.

    input_shape: (timesteps, features)
    """
    model = keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            keras.layers.LSTM(64, return_sequences=False),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(1),
        ]
    )

    model.compile(
        loss="mse",
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        metrics=["mae"],
    )
    return model