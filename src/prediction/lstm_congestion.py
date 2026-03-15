"""
LSTM-based Congestion Prediction Model
Capstone Research: Intelligent Traffic Management System
Author: nkhalfe56-star | IIT Jodhpur

Predicts traffic congestion probability for 5, 10, 15 minute horizons.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
from typing import List, Tuple, Optional
import logging
import joblib

logger = logging.getLogger(__name__)


class TrafficDataset(Dataset):
    """PyTorch Dataset for traffic time series data."""

    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class CongestionLSTM(nn.Module):
    """
    Stacked LSTM model for multi-horizon traffic congestion prediction.

    Input features per timestep:
      - vehicle_count: number of vehicles detected
      - avg_speed: average speed (km/h)
      - occupancy: lane occupancy ratio [0, 1]
      - hour_sin / hour_cos: cyclical time encoding
      - day_of_week: one-hot encoded (7 dims)
      - is_holiday: binary flag
      - weather_code: encoded weather condition

    Output: congestion probability for [5min, 10min, 15min] horizons
    """

    def __init__(
        self,
        input_dim: int = 14,
        hidden_dim: int = 128,
        num_layers: int = 3,
        output_horizons: int = 3,
        dropout: float = 0.2,
    ):
        super(CongestionLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True,
            dropout=dropout,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_horizons),
            nn.Sigmoid(),  # Output: congestion probability [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)

        # Self-attention over LSTM outputs
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Use last timestep
        out = attn_out[:, -1, :]  # (batch, hidden_dim)
        return self.fc(out)  # (batch, output_horizons)


class CongestionPredictor:
    """High-level wrapper for training and inference."""

    HORIZONS = [5, 10, 15]  # minutes

    def __init__(
        self,
        seq_len: int = 60,  # 60 timesteps (1 hour at 1-min resolution)
        input_dim: int = 14,
        hidden_dim: int = 128,
        num_layers: int = 3,
        lr: float = 1e-3,
        device: str = "auto",
    ):
        self.seq_len = seq_len
        self.scaler = MinMaxScaler()

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = CongestionLSTM(input_dim, hidden_dim, num_layers).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.BCELoss()
        logger.info(f"CongestionPredictor initialized on {self.device}")

    def _prepare_sequences(self, data: np.ndarray, targets: np.ndarray):
        X, y = [], []
        for i in range(len(data) - self.seq_len):
            X.append(data[i : i + self.seq_len])
            y.append(targets[i + self.seq_len])
        return np.array(X), np.array(y)

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_cols: List[str],
        epochs: int = 50,
        batch_size: int = 32,
        val_split: float = 0.2,
    ) -> dict:
        """Train the LSTM model."""
        features = self.scaler.fit_transform(df[feature_cols].values)
        targets = df[target_cols].values

        X, y = self._prepare_sequences(features, targets)

        split = int(len(X) * (1 - val_split))
        train_ds = TrafficDataset(X[:split], y[:split])
        val_ds = TrafficDataset(X[split:], y[split:])
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size)

        history = {"train_loss": [], "val_loss": [], "val_mae": []}

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for seqs, lbls in train_dl:
                seqs, lbls = seqs.to(self.device), lbls.to(self.device)
                preds = self.model(seqs)
                loss = self.criterion(preds, lbls)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                train_loss += loss.item()

            # Validation
            self.model.eval()
            val_preds, val_true = [], []
            val_loss = 0
            with torch.no_grad():
                for seqs, lbls in val_dl:
                    seqs, lbls = seqs.to(self.device), lbls.to(self.device)
                    preds = self.model(seqs)
                    val_loss += self.criterion(preds, lbls).item()
                    val_preds.append(preds.cpu().numpy())
                    val_true.append(lbls.cpu().numpy())

            val_preds = np.concatenate(val_preds)
            val_true = np.concatenate(val_true)
            mae = mean_absolute_error(val_true, val_preds)

            history["train_loss"].append(train_loss / len(train_dl))
            history["val_loss"].append(val_loss / len(val_dl))
            history["val_mae"].append(mae)

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss/len(train_dl):.4f} | Val MAE: {mae:.4f}")

        return history

    def predict(self, recent_data: np.ndarray) -> dict:
        """
        Predict congestion probabilities for 5, 10, 15 min horizons.

        Args:
            recent_data: array of shape (seq_len, n_features)

        Returns:
            dict with keys '5min', '10min', '15min' -> probability [0,1]
        """
        self.model.eval()
        scaled = self.scaler.transform(recent_data)
        tensor = torch.FloatTensor(scaled).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.model(tensor).cpu().numpy()[0]
        return {
            f"{h}min": float(probs[i])
            for i, h in enumerate(self.HORIZONS)
        }

    def save(self, model_path: str, scaler_path: str):
        torch.save(self.model.state_dict(), model_path)
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Model saved: {model_path}, Scaler saved: {scaler_path}")

    def load(self, model_path: str, scaler_path: str):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.scaler = joblib.load(scaler_path)
        logger.info(f"Model loaded from {model_path}")
