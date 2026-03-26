from __future__ import annotations

from dataclasses import dataclass, asdict
from collections import deque
import json
import math
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# =========================================================
# Version configs
# =========================================================
@dataclass(frozen=True)
class PreprocConfig:
    name: str
    L: int
    H: int
    hidden: int
    dropout: float
    loss: str
    huber_beta: float

    feature_set: str          # "v1" | "v2" | "v3" | "v4"
    target_mode: str          # "close" | "logret"
    x_scaler: str             # "minmax" | "standard"
    y_scaler: str             # "minmax" | "standard"

    clip_q_lo: float
    clip_q_hi: float
    clip_enabled: bool

    replay: int
    online_steps: int
    batch_lr: float
    online_lr: float
    weight_decay: float
    startup_epochs: int
    daily_epochs: int
    online_head_only: bool

    # kept only for backward compatibility with older saved states
    calibrate: bool
    calib_a_clip: float


# -------------------------
# V1 (frozen)
# -------------------------
CFG_V1 = PreprocConfig(
    name="v1",
    L=3,
    H=1,
    hidden=64,
    dropout=0.001,
    loss="mse",
    huber_beta=1.0,

    feature_set="v1",
    target_mode="close",
    x_scaler="minmax",
    y_scaler="minmax",

    clip_q_lo=0.01,
    clip_q_hi=0.99,
    clip_enabled=False,

    replay=16,
    online_steps=1,
    batch_lr=1e-3,
    online_lr=2e-4,
    weight_decay=0.0,
    startup_epochs=20,
    daily_epochs=20,
    online_head_only=False,

    calibrate=False,
    calib_a_clip=2.0,
)

# -------------------------
# V2 (frozen)
# -------------------------
CFG_V2 = PreprocConfig(
    name="v2",
    L=3,
    H=1,
    hidden=64,
    dropout=0.001,
    loss="mse",
    huber_beta=1.0,

    feature_set="v2",
    target_mode="logret",
    x_scaler="standard",
    y_scaler="standard",

    clip_q_lo=0.01,
    clip_q_hi=0.99,
    clip_enabled=True,

    replay=16,
    online_steps=1,
    batch_lr=1e-3,
    online_lr=2e-4,
    weight_decay=0.0,
    startup_epochs=20,
    daily_epochs=20,
    online_head_only=False,

    calibrate=False,
    calib_a_clip=2.0,
)

# -------------------------
# V3 (frozen)
# -------------------------
CFG_V3 = PreprocConfig(
    name="v3",
    L=3,
    H=1,
    hidden=64,
    dropout=0.001,
    loss="mse",
    huber_beta=1.0,

    feature_set="v3",
    target_mode="logret",
    x_scaler="standard",
    y_scaler="standard",

    clip_q_lo=0.01,
    clip_q_hi=0.99,
    clip_enabled=True,

    replay=16,
    online_steps=1,
    batch_lr=1e-3,
    online_lr=2e-4,
    weight_decay=0.0,
    startup_epochs=20,
    daily_epochs=20,
    online_head_only=False,

    calibrate=False,
    calib_a_clip=2.0,
)

# -------------------------
# V3A (frozen soft-architecture test)
# -------------------------
CFG_V3A = PreprocConfig(
    name="v3a",
    L=5,
    H=1,
    hidden=32,
    dropout=0.01,
    loss="mse",
    huber_beta=1.0,

    feature_set="v3",
    target_mode="logret",
    x_scaler="standard",
    y_scaler="standard",

    clip_q_lo=0.01,
    clip_q_hi=0.99,
    clip_enabled=True,

    replay=16,
    online_steps=1,
    batch_lr=1e-3,
    online_lr=1e-4,
    weight_decay=0.0,
    startup_epochs=20,
    daily_epochs=20,
    online_head_only=False,

    calibrate=False,
    calib_a_clip=2.0,
)

# -------------------------
# V3B (new hard-architecture test)
# same feature set and learning pipeline as V3
# architecture only: single Bi-H.BLSTM + temporal attention + compact head
# -------------------------
CFG_V3B = PreprocConfig(
    name="v3b",
    L=3,
    H=1,
    hidden=64,
    dropout=0.001,
    loss="mse",
    huber_beta=1.0,

    feature_set="v3",
    target_mode="logret",
    x_scaler="standard",
    y_scaler="standard",

    clip_q_lo=0.01,
    clip_q_hi=0.99,
    clip_enabled=True,

    replay=16,
    online_steps=1,
    batch_lr=1e-3,
    online_lr=2e-4,
    weight_decay=0.0,
    startup_epochs=20,
    daily_epochs=20,
    online_head_only=False,

    calibrate=False,
    calib_a_clip=2.0,
)

# -------------------------
# V4 (integrated volatility-aware model)
# fast branch + stable branch + regime-aware updates
# -------------------------
CFG_V4 = PreprocConfig(
    name="v4",
    L=8,
    H=1,
    hidden=64,
    dropout=0.01,
    loss="huber",
    huber_beta=0.75,

    feature_set="v4",
    target_mode="logret",
    x_scaler="standard",
    y_scaler="standard",

    clip_q_lo=0.01,
    clip_q_hi=0.99,
    clip_enabled=True,

    replay=24,
    online_steps=2,
    batch_lr=8e-4,
    online_lr=2e-4,
    weight_decay=1e-5,
    startup_epochs=25,
    daily_epochs=20,
    online_head_only=False,

    calibrate=False,
    calib_a_clip=2.0,
)


# =========================================================
# Online statistics
# =========================================================
@dataclass
class ResidualStats:
    maxlen: int = 500
    alpha: float = 0.05
    abs_hist: deque = None
    ewma_mu: float = 0.0
    ewma_var: float = 0.0
    n: int = 0

    def __post_init__(self):
        if self.abs_hist is None:
            self.abs_hist = deque(maxlen=self.maxlen)

    def update(self, resid_ret: float):
        if not np.isfinite(resid_ret):
            return
        abs_resid = float(abs(resid_ret))
        self.abs_hist.append(abs_resid)

        self.n += 1
        mu_prev = self.ewma_mu
        self.ewma_mu = (1.0 - self.alpha) * self.ewma_mu + self.alpha * float(resid_ret)
        self.ewma_var = (1.0 - self.alpha) * self.ewma_var + self.alpha * (float(resid_ret) - mu_prev) ** 2

    def sigma(self) -> float:
        return float(math.sqrt(max(self.ewma_var, 0.0)))

    def q_abs(self, q: float) -> float:
        if len(self.abs_hist) < 30:
            return 0.0
        arr = np.asarray(list(self.abs_hist), dtype=float)
        return float(np.quantile(arr, q))


@dataclass
class PageHinkley:
    delta: float = 0.0
    lambd: float = 0.01
    mean: float = 0.0
    cum: float = 0.0
    min_cum: float = 0.0
    n: int = 0

    def update(self, x: float) -> bool:
        if not np.isfinite(x):
            return False
        self.n += 1
        self.mean += (x - self.mean) / self.n
        self.cum += (x - self.mean - self.delta)
        self.min_cum = min(self.min_cum, self.cum)
        return (self.cum - self.min_cum) > self.lambd

    def reset(self):
        self.mean = 0.0
        self.cum = 0.0
        self.min_cum = 0.0
        self.n = 0


# kept only for compatibility
@dataclass
class CalibratorState:
    n: int = 0
    sx: float = 0.0
    sy: float = 0.0
    sxx: float = 0.0
    sxy: float = 0.0
    a: float = 1.0
    b: float = 0.0
    last_calib_label_time: str | None = None


# =========================================================
# Model bundle
# =========================================================
@dataclass
class ModelBundle:
    cfg: PreprocConfig
    model: nn.Module
    x_scaler: object
    y_scaler: object
    opt_online: optim.Optimizer
    device: torch.device
    is_multi: bool

    y_clip_lo: float | None
    y_clip_hi: float | None
    y_clip_q: float | None
    ret_clip_q: float | None

    last_update_time: str | None = None

    last_pred_time: str | None = None
    last_pred_next_close: float | None = None
    last_pred_next_close_raw: float | None = None
    last_pred_lr_raw: float | None = None
    last_pred_lr_used: float | None = None

    resid_stats: ResidualStats = None
    ph: PageHinkley = None
    drift_flag: bool = False
    last_drift_time: str | None = None

    calib: CalibratorState | None = None

    v4_vol_threshold: float | None = None
    last_regime: str | None = None
    last_realized_vol: float | None = None
    last_vol_ratio: float | None = None
    last_gate_fast_weight: float | None = None
    last_online_mode: str | None = None


# =========================================================
# Feature helpers
# =========================================================
def _logret_1(close: np.ndarray) -> np.ndarray:
    lr = np.zeros_like(close, dtype=np.float64)
    lr[1:] = np.log(np.maximum(close[1:], 1e-12)) - np.log(np.maximum(close[:-1], 1e-12))
    return lr


def _sma(x: np.ndarray, win: int) -> np.ndarray:
    return pd.Series(x).rolling(win, min_periods=1).mean().to_numpy(dtype=np.float64)


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(x).ewm(span=span, adjust=False).mean().to_numpy(dtype=np.float64)


def _roll_std(x: np.ndarray, win: int) -> np.ndarray:
    return pd.Series(x, dtype=float).rolling(win, min_periods=2).std(ddof=0).fillna(0.0).to_numpy(dtype=np.float64)


def _rsi_wilder(close: np.ndarray, period: int = 5) -> np.ndarray:
    s = pd.Series(close, dtype=float)
    delta = s.diff()

    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(50.0).clip(lower=0.0, upper=100.0)
    return rsi.to_numpy(dtype=np.float64)


# =========================================================
# Feature engineering
# =========================================================
def add_features(df: pd.DataFrame, cfg: PreprocConfig) -> pd.DataFrame:
    df = df.copy()
    close = df["close"].astype(float).values

    # Shared base feature
    df["EMA5"] = pd.Series(close).ewm(span=5, adjust=False).mean().to_numpy(dtype=np.float64)

    # V2/V4 engineered block
    df["hl_range"] = (df["high"] - df["low"]) / np.maximum(df["close"], 1e-12)
    df["oc_return"] = (df["close"] - df["open"]) / np.maximum(df["open"], 1e-12)
    df["log_vol"] = np.log1p(np.maximum(df["volume"].astype(float).values, 0.0))
    df["logret_1"] = _logret_1(close)

    # V3/V3A/V3B/V4 indicator block
    if cfg.feature_set in {"v3", "v4"}:
        ema5 = _ema(close, 5)
        ema10 = _ema(close, 10)
        ma5 = _sma(close, 5)
        macd5 = ema5 - ema10
        rsi5 = _rsi_wilder(close, period=5)

        df["MA5"] = ma5
        df["MACD5"] = macd5
        df["RSI5"] = rsi5

    # V4 volatility-aware block
    if cfg.feature_set == "v4":
        lr1 = df["logret_1"].astype(float).values
        rv8 = _roll_std(lr1, 8)
        rv32 = _roll_std(lr1, 32)
        vol_ratio = rv8 / np.maximum(rv32, 1e-8)
        mom5 = pd.Series(lr1, dtype=float).rolling(5, min_periods=1).mean().to_numpy(dtype=np.float64)
        mom_z = mom5 / np.maximum(rv8, 1e-8)

        df["RV8"] = rv8
        df["RV32"] = rv32
        df["vol_ratio"] = vol_ratio
        df["mom_z"] = mom_z

    return df


def make_X(df: pd.DataFrame, is_multi: bool, cfg: PreprocConfig) -> np.ndarray:
    df = add_features(df, cfg)

    if cfg.feature_set == "v1":
        feats = ["close", "EMA5"] if is_multi else ["close"]
        return df[feats].astype(float).values.astype(np.float32)

    if cfg.feature_set == "v2":
        base = ["close", "EMA5"] if is_multi else ["close"]
        extra_v2 = ["hl_range", "oc_return", "log_vol", "logret_1"]
        feats = base + extra_v2
        return df[feats].astype(float).values.astype(np.float32)

    if cfg.feature_set == "v3":
        feats = ["close", "EMA5", "MA5", "MACD5", "RSI5"]
        return df[feats].astype(float).values.astype(np.float32)

    if cfg.feature_set == "v4":
        feats = [
            "close",
            "EMA5",
            "MA5",
            "MACD5",
            "RSI5",
            "logret_1",
            "hl_range",
            "oc_return",
            "log_vol",
            "RV8",
            "RV32",
            "vol_ratio",
            "mom_z",
        ]
        return df[feats].astype(float).values.astype(np.float32)

    raise ValueError(f"Unknown feature_set: {cfg.feature_set}")


def input_dim(is_multi: bool, cfg: PreprocConfig) -> int:
    if cfg.feature_set == "v1":
        return 2 if is_multi else 1
    if cfg.feature_set == "v2":
        base = 2 if is_multi else 1
        return base + 4
    if cfg.feature_set == "v3":
        return 5
    if cfg.feature_set == "v4":
        return 13
    raise ValueError(f"Unknown feature_set: {cfg.feature_set}")



# =========================================================
# V4 regime helpers
# =========================================================
def compute_v4_state(df: pd.DataFrame, threshold: float | None = None) -> tuple[float, float, float, str]:
    feat_df = add_features(df, CFG_V4)
    rv = float(feat_df["RV8"].iloc[-1]) if "RV8" in feat_df.columns and len(feat_df) else 0.0
    vol_ratio = float(feat_df["vol_ratio"].iloc[-1]) if "vol_ratio" in feat_df.columns and len(feat_df) else 1.0

    if threshold is None:
        arr = feat_df["RV8"].astype(float).values if "RV8" in feat_df.columns else np.array([0.0], dtype=float)
        threshold = float(np.quantile(arr, 0.65)) if len(arr) else 0.0

    threshold = float(max(threshold, 1e-8))
    regime = "high" if rv >= threshold else "low"
    return rv, vol_ratio, threshold, regime


def compute_v4_threshold(df: pd.DataFrame) -> float:
    feat_df = add_features(df, CFG_V4)
    if "RV8" not in feat_df.columns or len(feat_df) < 20:
        return 1e-6
    thr = float(np.quantile(feat_df["RV8"].astype(float).values, 0.65))
    return float(max(thr, 1e-6))


# =========================================================
# Targets
# =========================================================
def make_y(df: pd.DataFrame, cfg: PreprocConfig) -> np.ndarray:
    close = df["close"].astype(float).values

    if cfg.target_mode == "close":
        return close.reshape(-1, 1).astype(np.float32)

    lr1 = _logret_1(close).astype(np.float32)

    if cfg.target_mode == "logret":
        return lr1.reshape(-1, 1)

    raise ValueError(f"Unknown target_mode: {cfg.target_mode}")


def reconstruct_next_close(last_close: float, pred_logret: float) -> float:
    return float(last_close * np.exp(pred_logret))


# =========================================================
# Sequences
# =========================================================
def make_sequences(X: np.ndarray, y: np.ndarray, close: np.ndarray, cfg: PreprocConfig):
    L, H = cfg.L, cfg.H
    X_seq, y_seq, last_close_seq = [], [], []
    max_i = len(X) - (L - 1) - H
    if max_i <= 0:
        return (
            np.zeros((0, L, X.shape[1]), dtype=np.float32),
            np.zeros((0, 1), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    for i in range(max_i):
        idx = (i + L - 1) + H
        X_seq.append(X[i:i + L])
        y_seq.append(y[idx])
        last_close_seq.append(float(close[idx - 1]))

    return (
        np.array(X_seq, dtype=np.float32),
        np.array(y_seq, dtype=np.float32),
        np.array(last_close_seq, dtype=np.float32),
    )


# =========================================================
# H.BLSTM cells
# =========================================================
class CustomHBLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.W_f = nn.Linear(input_size, hidden_size)
        self.U_f = nn.Linear(hidden_size, hidden_size)
        self.W_i = nn.Linear(input_size, hidden_size)
        self.U_i = nn.Linear(hidden_size, hidden_size)
        self.W_o = nn.Linear(input_size, hidden_size)
        self.U_o = nn.Linear(hidden_size, hidden_size)
        self.W_u = nn.Linear(input_size, hidden_size)
        self.U_u = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(input_size, hidden_size)
        self.U_k = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, hidden):
        h_prev, k_prev, kp_prev = hidden
        f_gate = torch.sigmoid(self.W_f(x) + self.U_f(h_prev))
        i_gate = torch.sigmoid(self.W_i(x) + self.U_i(h_prev))
        o_gate = torch.sigmoid(self.W_o(x) + self.U_o(h_prev))
        u_gate = torch.sigmoid(self.W_u(x) + self.U_u(h_prev))
        kp = u_gate * torch.tanh(self.W_k(x) + self.U_k(h_prev)) + (1.0 - u_gate) * kp_prev
        k_state = f_gate * k_prev + i_gate * kp
        h_state = o_gate * torch.tanh(k_state)
        return h_state, k_state, kp


class _BiHBLSTMLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout_rate: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.fwd = CustomHBLSTMCell(input_size, hidden_size)
        self.bwd = CustomHBLSTMCell(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def _zeros(self, batch_size: int, device: torch.device):
        return (
            torch.zeros(batch_size, self.hidden_size, device=device),
            torch.zeros(batch_size, self.hidden_size, device=device),
            torch.zeros(batch_size, self.hidden_size, device=device),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        h, k, kp = self._zeros(batch_size, x.device)
        out_f = []
        for t in range(seq_len):
            h, k, kp = self.fwd(x[:, t, :], (h, k, kp))
            out_f.append(h)
        out_f = torch.stack(out_f, dim=1)

        h, k, kp = self._zeros(batch_size, x.device)
        out_b = []
        for t in reversed(range(seq_len)):
            h, k, kp = self.bwd(x[:, t, :], (h, k, kp))
            out_b.insert(0, h)
        out_b = torch.stack(out_b, dim=1)

        seq = torch.cat([out_f, out_b], dim=2)
        return self.dropout(seq)


class H_BLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, dropout_rate: float = 0.001):
        super().__init__()
        self.hidden_size = hidden_size
        self.f1 = CustomHBLSTMCell(input_size, hidden_size)
        self.b1 = CustomHBLSTMCell(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.f2 = CustomHBLSTMCell(hidden_size * 2, hidden_size)
        self.b2 = CustomHBLSTMCell(hidden_size * 2, hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def _zeros(self, batch_size: int, device: torch.device):
        return (
            torch.zeros(batch_size, self.hidden_size, device=device),
            torch.zeros(batch_size, self.hidden_size, device=device),
            torch.zeros(batch_size, self.hidden_size, device=device),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        h, k, kp = self._zeros(batch_size, x.device)
        out_f = []
        for t in range(seq_len):
            h, k, kp = self.f1(x[:, t, :], (h, k, kp))
            out_f.append(h)
        out_f = torch.stack(out_f, dim=1)

        h, k, kp = self._zeros(batch_size, x.device)
        out_b = []
        for t in reversed(range(seq_len)):
            h, k, kp = self.b1(x[:, t, :], (h, k, kp))
            out_b.insert(0, h)
        out_b = torch.stack(out_b, dim=1)

        layer1 = torch.cat([out_f, out_b], dim=2)
        layer1 = self.dropout(layer1)

        h, k, kp = self._zeros(batch_size, x.device)
        for t in range(seq_len):
            h, k, kp = self.f2(layer1[:, t, :], (h, k, kp))
        hf = h

        h, k, kp = self._zeros(batch_size, x.device)
        for t in reversed(range(seq_len)):
            h, k, kp = self.b2(layer1[:, t, :], (h, k, kp))
        hb = h

        return self.head(torch.cat([hf, hb], dim=1))


class HBLSTMTemporalAttention(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, dropout_rate: float = 0.001):
        super().__init__()
        self.encoder = _BiHBLSTMLayer(input_size=input_size, hidden_size=hidden_size, dropout_rate=dropout_rate)
        self.attn_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.attn_score = nn.Linear(hidden_size, 1)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = self.encoder(x)  # (B, L, 2H)
        attn_hidden = torch.tanh(self.attn_proj(seq))
        attn_logits = self.attn_score(attn_hidden).squeeze(-1)  # (B, L)
        attn_weights = torch.softmax(attn_logits, dim=1).unsqueeze(-1)  # (B, L, 1)
        context = torch.sum(seq * attn_weights, dim=1)  # (B, 2H)
        return self.head(context)



class VolatilityAwareDualPath(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, dropout_rate: float = 0.01):
        super().__init__()
        self.fast_branch = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.stable_branch = _BiHBLSTMLayer(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
        )
        self.fast_proj = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.stable_proj = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.attn_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.attn_score = nn.Linear(hidden_size, 1)
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 2, 1),
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 6, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 2, 1),
        )
        self.last_gate_mean: float | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fast_seq, _ = self.fast_branch(x)
        fast_last = self.fast_proj(fast_seq[:, -1, :])

        stable_seq = self.stable_branch(x)
        attn_hidden = torch.tanh(self.attn_proj(stable_seq))
        attn_logits = self.attn_score(attn_hidden).squeeze(-1)
        attn_weights = torch.softmax(attn_logits, dim=1).unsqueeze(-1)
        stable_ctx = torch.sum(stable_seq * attn_weights, dim=1)
        stable_ctx = self.stable_proj(stable_ctx)

        gate = torch.sigmoid(self.gate(torch.cat([fast_last, stable_ctx], dim=1)))
        fused = gate * fast_last + (1.0 - gate) * stable_ctx

        try:
            self.last_gate_mean = float(gate.detach().mean().cpu().item())
        except Exception:
            self.last_gate_mean = None

        out = torch.cat([fused, fast_last, stable_ctx], dim=1)
        return self.head(self.dropout(out))


def build_model(cfg: PreprocConfig, is_multi: bool) -> nn.Module:
    in_dim = input_dim(is_multi, cfg)
    if cfg.name == "v3b":
        return HBLSTMTemporalAttention(input_size=in_dim, hidden_size=cfg.hidden, dropout_rate=cfg.dropout)
    if cfg.name == "v4":
        return VolatilityAwareDualPath(input_size=in_dim, hidden_size=cfg.hidden, dropout_rate=cfg.dropout)
    return H_BLSTM(input_size=in_dim, hidden_size=cfg.hidden, dropout_rate=cfg.dropout)


# =========================================================
# Scaling + clipping
# =========================================================
def _make_scaler(kind: str):
    if kind == "standard":
        return StandardScaler()
    if kind == "minmax":
        return MinMaxScaler(feature_range=(0.0, 1.0))
    raise ValueError(f"Unknown scaler type: {kind}")


def fit_scalers(X_seq: np.ndarray, y_seq: np.ndarray, cfg: PreprocConfig):
    dim = X_seq.shape[2]
    X2 = X_seq.reshape(-1, dim)
    xs = _make_scaler(cfg.x_scaler).fit(X2)
    ys = _make_scaler(cfg.y_scaler).fit(y_seq)
    return xs, ys


def scale_X(xs, X_seq: np.ndarray) -> np.ndarray:
    dim = X_seq.shape[2]
    X2 = X_seq.reshape(-1, dim)
    X2s = xs.transform(X2)
    return X2s.reshape(X_seq.shape).astype(np.float32)


def scale_y(ys, y_seq: np.ndarray) -> np.ndarray:
    return ys.transform(y_seq).astype(np.float32)


def inv_y(ys, y_scaled: np.ndarray) -> np.ndarray:
    return ys.inverse_transform(y_scaled)


def fit_clip_bounds(y_train: np.ndarray, cfg: PreprocConfig):
    lo = float(np.quantile(y_train.reshape(-1), cfg.clip_q_lo))
    hi = float(np.quantile(y_train.reshape(-1), cfg.clip_q_hi))
    return lo, hi


def symmetric_q(lo: float, hi: float) -> float:
    return float(max(abs(lo), abs(hi), 1e-12))


# =========================================================
# Loss
# =========================================================
def make_loss(cfg: PreprocConfig):
    if cfg.loss == "mse":
        return nn.MSELoss()
    if cfg.loss == "huber":
        return nn.SmoothL1Loss(beta=cfg.huber_beta)
    raise ValueError(f"Unknown loss: {cfg.loss}")


# =========================================================
# Return clip bound
# =========================================================
def compute_ret_clip_q(df: pd.DataFrame, cfg: PreprocConfig) -> float | None:
    close = df["close"].astype(float).values
    lr1 = _logret_1(close)
    if len(lr1) < 50:
        return None
    lo = float(np.quantile(lr1, cfg.clip_q_lo))
    hi = float(np.quantile(lr1, cfg.clip_q_hi))
    return symmetric_q(lo, hi)


# =========================================================
# Compatibility no-op
# =========================================================
def update_calibration(bundle: ModelBundle, label_time_iso: str, actual_lr: float):
    return


def apply_calibration(bundle: ModelBundle, lr_raw: float) -> float:
    return lr_raw


# =========================================================
# Train / Predict / Online update
# =========================================================
def train_batch(
    df_window: pd.DataFrame,
    device: torch.device,
    is_multi: bool,
    cfg: PreprocConfig,
    epochs: int | None = None,
) -> ModelBundle:
    df_window = df_window.copy().reset_index(drop=True)

    X_raw = make_X(df_window, is_multi=is_multi, cfg=cfg)
    y_raw = make_y(df_window, cfg=cfg)
    close = df_window["close"].astype(float).values

    X_seq, y_seq, _ = make_sequences(X_raw, y_raw, close, cfg)
    if len(X_seq) < max(cfg.replay + 10, 50):
        raise ValueError(f"Not enough sequences to train: got {len(X_seq)}")

    split = int(len(X_seq) * 0.9)
    split = max(split, max(cfg.replay + 1, 10))
    Xtr, ytr = X_seq[:split], y_seq[:split]

    y_clip_lo = y_clip_hi = y_clip_q = None
    if cfg.clip_enabled:
        y_clip_lo, y_clip_hi = fit_clip_bounds(ytr, cfg)
        y_clip_q = symmetric_q(y_clip_lo, y_clip_hi)
        ytr = np.clip(ytr, -y_clip_q, y_clip_q).astype(np.float32)

    xs, ys = fit_scalers(Xtr, ytr, cfg)
    Xtr_s = scale_X(xs, Xtr)
    ytr_s = scale_y(ys, ytr)

    model = build_model(cfg=cfg, is_multi=is_multi).to(device)

    opt = optim.Adam(model.parameters(), lr=cfg.batch_lr, weight_decay=cfg.weight_decay)
    loss_fn = make_loss(cfg)

    Xtr_t = torch.tensor(Xtr_s, device=device)
    ytr_t = torch.tensor(ytr_s, device=device)

    n_epochs = int(epochs if epochs is not None else cfg.startup_epochs)
    model.train()
    for _ in range(n_epochs):
        for i in range(0, len(Xtr_t), 64):
            bx = Xtr_t[i:i + 64]
            by = ytr_t[i:i + 64]
            opt.zero_grad()
            pred = model(bx)
            loss = loss_fn(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    if cfg.online_head_only and hasattr(model, "head"):
        params = list(model.head.parameters())
    else:
        params = list(model.parameters())
    opt_online = optim.Adam(params, lr=cfg.online_lr, weight_decay=cfg.weight_decay)

    ret_clip_q = compute_ret_clip_q(df_window.iloc[:split + cfg.L + cfg.H], cfg)
    v4_vol_threshold = None
    if cfg.name == "v4":
        train_slice = df_window.iloc[: split + cfg.L + cfg.H].copy().reset_index(drop=True)
        v4_vol_threshold = compute_v4_threshold(train_slice)

    return ModelBundle(
        cfg=cfg,
        model=model,
        x_scaler=xs,
        y_scaler=ys,
        opt_online=opt_online,
        device=device,
        is_multi=is_multi,
        y_clip_lo=y_clip_lo,
        y_clip_hi=y_clip_hi,
        y_clip_q=y_clip_q,
        ret_clip_q=ret_clip_q,
        last_update_time=None,
        resid_stats=ResidualStats(),
        ph=PageHinkley(),
        drift_flag=False,
        last_drift_time=None,
        calib=None,
        v4_vol_threshold=v4_vol_threshold,
        last_regime=None,
        last_realized_vol=None,
        last_vol_ratio=None,
        last_gate_fast_weight=None,
        last_online_mode=None,
    )


def predict_next(bundle: ModelBundle, df_buffer: pd.DataFrame) -> tuple[float, float, float, float]:
    cfg = bundle.cfg
    df_buffer = df_buffer.copy().reset_index(drop=True)

    if cfg.name == "v4":
        rv, vol_ratio, thr, regime = compute_v4_state(df_buffer, threshold=bundle.v4_vol_threshold)
        bundle.v4_vol_threshold = thr
        bundle.last_regime = regime
        bundle.last_realized_vol = rv
        bundle.last_vol_ratio = vol_ratio
    else:
        bundle.last_regime = None
        bundle.last_realized_vol = None
        bundle.last_vol_ratio = None
        bundle.last_gate_fast_weight = None
        bundle.last_online_mode = None

    close = df_buffer["close"].astype(float).values
    last_close = float(close[-1])
    base_time_iso = pd.to_datetime(df_buffer["time"].iloc[-1], utc=True).isoformat()

    pred_close_used = last_close
    pred_close_raw = last_close
    pred_lr_raw = 0.0
    pred_lr_used = 0.0

    if len(df_buffer) < cfg.L + 2:
        q95 = bundle.resid_stats.q_abs(0.95) if bundle.resid_stats else 0.0
        pi_lo = last_close * math.exp(-q95) if q95 > 0 else last_close
        pi_hi = last_close * math.exp(+q95) if q95 > 0 else last_close
        return pred_close_used, pred_close_raw, float(pi_lo), float(pi_hi)

    X_raw = make_X(df_buffer, is_multi=bundle.is_multi, cfg=cfg)
    X_win = X_raw[-cfg.L:].reshape(1, cfg.L, -1).astype(np.float32)
    X_win_s = scale_X(bundle.x_scaler, X_win)

    try:
        x_t = torch.tensor(X_win_s, device=bundle.device)
        bundle.model.eval()
        with torch.no_grad():
            p_s = bundle.model(x_t).detach().cpu().numpy()

        if cfg.name == "v4":
            bundle.last_gate_fast_weight = getattr(bundle.model, "last_gate_mean", None)

        pred_y = float(inv_y(bundle.y_scaler, p_s)[0, 0])

        if cfg.clip_enabled and bundle.y_clip_q is not None:
            pred_y = float(np.clip(pred_y, -bundle.y_clip_q, bundle.y_clip_q))

        if cfg.target_mode == "close":
            pred_close_raw = max(float(pred_y), 1e-6)
            pred_close_used = pred_close_raw
            pred_lr_raw = float(np.log((pred_close_raw + 1e-12) / (last_close + 1e-12)))
            pred_lr_used = pred_lr_raw
        elif cfg.target_mode == "logret":
            pred_lr_raw = float(pred_y)
            pred_lr_used = pred_lr_raw
        else:
            raise ValueError("Unknown target_mode")

        resid_q95 = bundle.resid_stats.q_abs(0.95) if bundle.resid_stats else 0.0
        qret = max(bundle.ret_clip_q or 0.0, resid_q95, 1e-12)
        pred_lr_raw = float(np.clip(pred_lr_raw, -qret, qret))
        pred_lr_used = float(np.clip(pred_lr_used, -qret, qret))

        if cfg.target_mode != "close":
            pred_close_raw = reconstruct_next_close(last_close, pred_lr_raw)
            pred_close_used = reconstruct_next_close(last_close, pred_lr_used)

        q95 = bundle.resid_stats.q_abs(0.95) if bundle.resid_stats else 0.0
        sigma = bundle.resid_stats.sigma() if bundle.resid_stats else 0.0
        width = max(q95, 1.96 * sigma)

        pi_lo = last_close * math.exp(pred_lr_used - width)
        pi_hi = last_close * math.exp(pred_lr_used + width)

        bundle.last_pred_time = base_time_iso
        bundle.last_pred_next_close = float(pred_close_used)
        bundle.last_pred_next_close_raw = float(pred_close_raw)
        bundle.last_pred_lr_raw = float(pred_lr_raw)
        bundle.last_pred_lr_used = float(pred_lr_used)

        return float(pred_close_used), float(pred_close_raw), float(pi_lo), float(pi_hi)

    except Exception:
        q95 = bundle.resid_stats.q_abs(0.95) if bundle.resid_stats else 0.0
        pi_lo = last_close * math.exp(-q95) if q95 > 0 else last_close
        pi_hi = last_close * math.exp(+q95) if q95 > 0 else last_close
        return last_close, last_close, float(pi_lo), float(pi_hi)


def online_update(bundle: ModelBundle, df_buffer: pd.DataFrame):
    cfg = bundle.cfg
    df_buffer = df_buffer.copy().reset_index(drop=True)

    label_time_iso = pd.to_datetime(df_buffer["time"].iloc[-1], utc=True).isoformat()
    if bundle.last_update_time is not None and label_time_iso <= bundle.last_update_time:
        return

    X_raw = make_X(df_buffer, is_multi=bundle.is_multi, cfg=cfg)
    y_raw = make_y(df_buffer, cfg=cfg)
    close = df_buffer["close"].astype(float).values

    X_seq, y_seq, _ = make_sequences(X_raw, y_raw, close, cfg)
    if len(X_seq) < cfg.replay:
        bundle.last_update_time = label_time_iso
        return

    replay = int(cfg.replay)
    online_steps = int(cfg.online_steps)

    if cfg.name == "v4":
        rv, vol_ratio, thr, regime = compute_v4_state(df_buffer, threshold=bundle.v4_vol_threshold)
        bundle.v4_vol_threshold = thr
        bundle.last_regime = regime
        bundle.last_realized_vol = rv
        bundle.last_vol_ratio = vol_ratio

        if regime == "high":
            replay = min(len(X_seq), max(cfg.replay, 32))
            online_steps = max(cfg.online_steps, 3)
            bundle.last_online_mode = "aggressive_recent"
        else:
            replay = min(len(X_seq), max(12, cfg.replay // 2))
            online_steps = max(1, cfg.online_steps)
            bundle.last_online_mode = "stable_conservative"
    else:
        bundle.last_online_mode = None

    Xb = X_seq[-replay:]
    yb = y_seq[-replay:]

    if cfg.clip_enabled and bundle.y_clip_q is not None:
        yb = np.clip(yb, -bundle.y_clip_q, bundle.y_clip_q).astype(np.float32)

    Xb_s = scale_X(bundle.x_scaler, Xb)
    yb_s = scale_y(bundle.y_scaler, yb)

    xb_t = torch.tensor(Xb_s, device=bundle.device)
    yb_t = torch.tensor(yb_s, device=bundle.device)

    bundle.model.train()
    loss_fn = make_loss(cfg)

    if cfg.online_head_only and hasattr(bundle.model, "head"):
        for p in bundle.model.parameters():
            p.requires_grad = False
        for p in bundle.model.head.parameters():
            p.requires_grad = True

    for _ in range(online_steps):
        bundle.opt_online.zero_grad()
        pred = bundle.model(xb_t)
        loss = loss_fn(pred, yb_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bundle.model.parameters(), 1.0)
        bundle.opt_online.step()

    if cfg.online_head_only and hasattr(bundle.model, "head"):
        for p in bundle.model.parameters():
            p.requires_grad = True

    bundle.last_update_time = label_time_iso


# =========================================================
# Checkpointing
# =========================================================
def cfg_dict(cfg: PreprocConfig) -> dict:
    return asdict(cfg)


def save_bundle(bundle: ModelBundle, ckpt_dir: str, meta: dict):
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(bundle.model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
    with open(os.path.join(ckpt_dir, "state.pkl"), "wb") as f:
        pickle.dump({
            "cfg": cfg_dict(bundle.cfg),
            "is_multi": bundle.is_multi,
            "x_scaler": bundle.x_scaler,
            "y_scaler": bundle.y_scaler,
            "y_clip_lo": bundle.y_clip_lo,
            "y_clip_hi": bundle.y_clip_hi,
            "y_clip_q": bundle.y_clip_q,
            "ret_clip_q": bundle.ret_clip_q,
            "last_update_time": bundle.last_update_time,
            "last_pred_time": bundle.last_pred_time,
            "last_pred_next_close": bundle.last_pred_next_close,
            "last_pred_next_close_raw": bundle.last_pred_next_close_raw,
            "last_pred_lr_raw": bundle.last_pred_lr_raw,
            "last_pred_lr_used": bundle.last_pred_lr_used,
            "resid_stats": bundle.resid_stats,
            "ph": bundle.ph,
            "drift_flag": bundle.drift_flag,
            "last_drift_time": bundle.last_drift_time,
            "calib": bundle.calib,
            "v4_vol_threshold": bundle.v4_vol_threshold,
            "last_regime": bundle.last_regime,
            "last_realized_vol": bundle.last_realized_vol,
            "last_vol_ratio": bundle.last_vol_ratio,
            "last_gate_fast_weight": bundle.last_gate_fast_weight,
            "last_online_mode": bundle.last_online_mode,
        }, f)
    with open(os.path.join(ckpt_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)


def load_bundle(ckpt_dir: str, device: torch.device, is_multi: bool, cfg: PreprocConfig) -> ModelBundle | None:
    model_path = os.path.join(ckpt_dir, "model.pt")
    state_path = os.path.join(ckpt_dir, "state.pkl")
    if not (os.path.exists(model_path) and os.path.exists(state_path)):
        return None

    with open(state_path, "rb") as f:
        st = pickle.load(f)

    if st.get("cfg") != cfg_dict(cfg):
        return None
    if st.get("is_multi") != is_multi:
        return None

    model = build_model(cfg=cfg, is_multi=is_multi).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    if cfg.online_head_only and hasattr(model, "head"):
        params = list(model.head.parameters())
    else:
        params = list(model.parameters())
    opt_online = optim.Adam(params, lr=cfg.online_lr, weight_decay=cfg.weight_decay)

    return ModelBundle(
        cfg=cfg,
        model=model,
        x_scaler=st["x_scaler"],
        y_scaler=st["y_scaler"],
        opt_online=opt_online,
        device=device,
        is_multi=is_multi,
        y_clip_lo=st.get("y_clip_lo"),
        y_clip_hi=st.get("y_clip_hi"),
        y_clip_q=st.get("y_clip_q"),
        ret_clip_q=st.get("ret_clip_q"),
        last_update_time=st.get("last_update_time"),
        last_pred_time=st.get("last_pred_time"),
        last_pred_next_close=st.get("last_pred_next_close"),
        last_pred_next_close_raw=st.get("last_pred_next_close_raw"),
        last_pred_lr_raw=st.get("last_pred_lr_raw"),
        last_pred_lr_used=st.get("last_pred_lr_used"),
        resid_stats=st.get("resid_stats") or ResidualStats(),
        ph=st.get("ph") or PageHinkley(),
        drift_flag=bool(st.get("drift_flag", False)),
        last_drift_time=st.get("last_drift_time"),
        calib=st.get("calib"),
        v4_vol_threshold=st.get("v4_vol_threshold"),
        last_regime=st.get("last_regime"),
        last_realized_vol=st.get("last_realized_vol"),
        last_vol_ratio=st.get("last_vol_ratio"),
        last_gate_fast_weight=st.get("last_gate_fast_weight"),
        last_online_mode=st.get("last_online_mode"),
    )
