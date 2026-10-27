"""1D-convolutional autoencoder; anomaly score = reconstruction MSE."""
import numpy as np
import tensorflow as tf


def build_autoencoder(length: int, n_features: int, width: int = 32) -> tf.keras.Model:
    """Three conv layers down, three up. Uses only ops that TFLite Micro supports in INT8."""
    L = tf.keras.layers
    x_in = L.Input((length, n_features))
    h = L.Conv1D(width, 5, strides=2, padding="same", activation="relu")(x_in)
    h = L.Conv1D(width // 2, 5, strides=2, padding="same", activation="relu")(h)
    h = L.Conv1D(width // 4, 3, strides=1, padding="same", activation="relu")(h)  # bottleneck
    h = L.Conv1DTranspose(width // 2, 5, strides=2, padding="same", activation="relu")(h)
    h = L.Conv1DTranspose(width, 5, strides=2, padding="same", activation="relu")(h)
    x_out = L.Conv1D(n_features, 3, padding="same")(h)
    return tf.keras.Model(x_in, x_out)


def train(model: tf.keras.Model, x: np.ndarray, seed: int, epochs: int = 20,
          batch: int = 256) -> tf.keras.Model:
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    model.fit(x, x, epochs=epochs, batch_size=batch, validation_split=0.1, shuffle=True,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
              verbose=0)
    return model


def score_from_recon(x: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
    """Per-window reconstruction MSE. The firmware computes the same quantity."""
    return ((x - x_hat) ** 2).mean(axis=(1, 2))


def score_fp32(model: tf.keras.Model, x: np.ndarray, batch: int = 1024) -> np.ndarray:
    return score_from_recon(x, model.predict(x, batch_size=batch, verbose=0))
