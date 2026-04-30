"""
bot/lstm_model.py
─────────────────
LSTM (Long Short-Term Memory) neural network for cryptocurrency price prediction.

Architecture:
    Input  → LSTM(128) → Dropout(0.2) → BatchNorm
           → LSTM(64)  → Dropout(0.2) → BatchNorm
           → Dense(32, relu) → Dropout(0.1)
           → Dense(1)        → predicted next-candle close price (scaled)

Lookback window : 60 candles (60 hours of market history per prediction)
Input shape     : (batch, 60, 22) — 60 timesteps × 22 features
Output shape    : (batch, 1)      — next-candle scaled close price
"""

import logging
import os
import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join("models", "lstm_model.keras")
LOOKBACK   = 60    # Number of historical candles used per prediction


# ── Sequence builder ──────────────────────────────────────────────────────────

def build_sequences(X_scaled: np.ndarray, lookback: int = LOOKBACK) -> tuple:
    """
    Create sliding-window sequences for LSTM training.

    Input  : X_scaled shape (n_samples, n_features)
    Output : X shape (n_seq, lookback, n_features)
             y shape (n_seq,) — next-candle scaled 'close' (column 0)
    """
    X, y = [], []
    for i in range(lookback, len(X_scaled)):
        X.append(X_scaled[i - lookback : i])   # 60-candle window of all features
        y.append(X_scaled[i, 0])               # next candle's scaled 'close'
    logger.info(f"Built {len(X)} training sequences (lookback={lookback})")
    return np.array(X), np.array(y)


# ── Model builder ─────────────────────────────────────────────────────────────

def build_model(n_features: int):
    """Build and compile the stacked LSTM model."""
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    except ImportError:
        raise ImportError(
            "TensorFlow is required for the AI module.\n"
            "Install with: pip install -r requirements_ai.txt"
        )

    model = Sequential([
        # ── Layer 1: LSTM 128 units, return sequences for stacking ───────────
        LSTM(128, return_sequences=True, input_shape=(LOOKBACK, n_features)),
        Dropout(0.2),
        BatchNormalization(),

        # ── Layer 2: LSTM 64 units, summarise the full 60-hour pattern ───────
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        BatchNormalization(),

        # ── Dense regression head ─────────────────────────────────────────────
        Dense(32, activation="relu"),
        Dropout(0.1),
        Dense(1),   # single output: predicted next-candle scaled close price
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mean_squared_error",
        metrics=["mae"],
    )

    logger.info(f"LSTM model built | total parameters: {model.count_params():,}")
    return model


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(X_scaled: np.ndarray, epochs: int = 50, batch_size: int = 32):
    """
    Train LSTM on historical feature data and save weights to models/lstm_model.keras

    Args:
        X_scaled   : Normalized feature matrix from build_features()
        epochs     : Maximum training epochs (EarlyStopping may stop earlier)
        batch_size : Mini-batch size

    Returns:
        Trained tf.keras.Model
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    except ImportError:
        raise ImportError("Install TensorFlow: pip install -r requirements_ai.txt")

    X, y = build_sequences(X_scaled)

    # Chronological split — no shuffle (time-series data)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15, shuffle=False)

    n_features = X.shape[2]
    model      = build_model(n_features)

    callbacks = [
        # Stop if val_loss stops improving for 8 epochs
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True,
                      verbose=1),
        # Halve LR if val_loss plateaus for 4 epochs
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4,
                          min_lr=1e-6, verbose=1),
    ]

    logger.info(f"Training LSTM | X_train={X_tr.shape} X_val={X_val.shape}")
    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Save model
    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)
    logger.info(f"Model saved → {MODEL_PATH}")
    return model


# ── Prediction ────────────────────────────────────────────────────────────────

def predict_next_close(X_scaled: np.ndarray, close_scaler) -> float:
    """
    Load saved LSTM model and predict the next-candle closing price.

    Args:
        X_scaled     : Full normalized feature matrix (only last 60 rows are used)
        close_scaler : MinMaxScaler fitted on 'close' column (for inverse-transform)

    Returns:
        predicted_price : float — predicted next-candle close in USDT
    """
    try:
        from tensorflow.keras.models import load_model as keras_load
    except ImportError:
        raise ImportError("Install TensorFlow: pip install -r requirements_ai.txt")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            f"Run: python ai_trader.py --train"
        )

    model     = keras_load(MODEL_PATH)
    last_seq  = X_scaled[-LOOKBACK:].reshape(1, LOOKBACK, X_scaled.shape[1])
    pred_sc   = model.predict(last_seq, verbose=0)[0][0]

    # Inverse-transform from [0,1] back to real USDT price
    predicted = close_scaler.inverse_transform([[pred_sc]])[0][0]
    logger.info(f"LSTM prediction: {predicted:.2f} USDT (scaled={pred_sc:.6f})")
    return float(predicted)
