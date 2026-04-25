# app.py  (RUNS V1 + V2 + V3 + V3A + V3B across 15m + 30m + 1h)
# V1, V2, V3, V3A are frozen.
# V3B is the new hard-architecture comparison:
#   same features as V3, same target/scaling pipeline,
#   architecture only changed to single Bi-H.BLSTM + temporal attention.

import argparse
import os
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from coinbase import bootstrap_last_n_public, fetch_latest_public, merge_roll
from model import (
    CFG_V1,
    CFG_V2,
    CFG_V3,
    CFG_V3A,
    CFG_V3B,
    load_bundle,
    online_update,
    predict_next,
    save_bundle,
    train_batch,
    update_calibration,
)


# =========================
# USER CONFIG
# =========================
PRODUCT_IDS = ["BTC-USD", "ETH-USD", "SOL-USD"]
IS_MULTI = True
SAVE_EVERY_N_PRED = 1
DAILY_RETRAIN_HOURS = 24

# Base wake-up interval = 15 minutes
BASE_ALIGN_SECONDS = 15 * 60

# Timeframes to run in parallel
TIMEFRAMES = {
    "15m": {
        "granularity": "FIFTEEN_MINUTE",
        "seconds": 15 * 60,
        "window": 6000,
    },
    "30m": {
        "granularity": "THIRTY_MINUTE",
        "seconds": 30 * 60,
        "window": 6000,
    },
    "1h": {
        "granularity": "ONE_HOUR",
        "seconds": 60 * 60,
        "window": 6000,
    },
}

# Drift retrain disabled to avoid changing frozen versions
DRIFT_RETRAIN_ENABLED = False
DRIFT_RETRAIN_COOLDOWN_HOURS = 2

# roots
RAW_ROOT = "data_raw"
OUT_ROOTS = {
    "v1": "data_out_v1",
    "v2": "data_out_v2",
    "v3": "data_out_v3",
    "v3a": "data_out_v3a",
    "v3b": "data_out_v3b",
}
CKPT_ROOTS = {
    "v1": "checkpoints_v1",
    "v2": "checkpoints_v2",
    "v3": "checkpoints_v3",
    "v3a": "checkpoints_v3a",
    "v3b": "checkpoints_v3b",
}

CONFIGS = {
    "v1": CFG_V1,
    "v2": CFG_V2,
    "v3": CFG_V3,
    "v3a": CFG_V3A,
    "v3b": CFG_V3B,
}
VERSIONS_ORDER = ["v1", "v2", "v3", "v3a", "v3b"]


# =========================
# Timezone policy
# Keep all internal scheduling, storage, and joins in UTC.
# Only logs / dashboard presentation are shown in Europe/London.
# =========================
DISPLAY_TZ = ZoneInfo("Europe/London")


def now_utc():
    return datetime.now(timezone.utc)


def now_local() -> datetime:
    return now_utc().astimezone(DISPLAY_TZ)


def fmt_local(ts) -> str:
    if ts is None:
        return ""
    return pd.to_datetime(ts, utc=True).tz_convert(DISPLAY_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")


def fmt_dual(ts) -> str:
    ts_utc = pd.to_datetime(ts, utc=True)
    ts_local = ts_utc.tz_convert(DISPLAY_TZ)
    return f"{ts_local.strftime('%Y-%m-%d %H:%M:%S %Z')} | {ts_utc.isoformat()}"


# =========================
# Path helpers
# =========================
def raw_dir(tf_name: str, pid: str) -> str:
    return os.path.join(RAW_ROOT, tf_name, pid)


def raw_path(tf_name: str, pid: str) -> str:
    return os.path.join(raw_dir(tf_name, pid), "buffer.parquet")


def out_dir(ver: str, tf_name: str, pid: str) -> str:
    return os.path.join(OUT_ROOTS[ver], tf_name, pid)


def ckpt_dir(ver: str, tf_name: str, pid: str) -> str:
    return os.path.join(CKPT_ROOTS[ver], tf_name, pid)


def out_pred_path(ver: str, tf_name: str, pid: str) -> str:
    return os.path.join(out_dir(ver, tf_name, pid), f"predictions_{ver}.csv")


def out_log_path(ver: str, tf_name: str, pid: str) -> str:
    return os.path.join(out_dir(ver, tf_name, pid), f"log_{ver}.txt")


def ensure_dirs():
    for tf_name in TIMEFRAMES:
        for pid in PRODUCT_IDS:
            os.makedirs(raw_dir(tf_name, pid), exist_ok=True)
            for ver in CONFIGS:
                os.makedirs(out_dir(ver, tf_name, pid), exist_ok=True)
                os.makedirs(ckpt_dir(ver, tf_name, pid), exist_ok=True)


def append_log(ver: str, tf_name: str, pid: str, line: str):
    with open(out_log_path(ver, tf_name, pid), "a", encoding="utf-8") as f:
        f.write(line + "\n")


# =========================
# Safe CSV append
# =========================
CSV_COLUMNS = [
    "time",
    "last_close",
    "pred_next_close",
    "pred_next_close_raw",
    "pi95_lower",
    "pi95_upper",
    "naive_next_close",
]


def _read_last_time_csv(csv_path: str) -> str | None:
    if not os.path.exists(csv_path):
        return None
    try:
        with open(csv_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size == 0:
                return None
            f.seek(max(size - 4096, 0), os.SEEK_SET)
            tail = f.read().decode("utf-8", errors="ignore").strip().splitlines()
            if len(tail) <= 1:
                return None
            return tail[-1].split(",")[0].strip()
    except Exception:
        return None


def append_prediction(
    ver: str,
    tf_name: str,
    pid: str,
    ts,
    last_close: float,
    pred_close: float,
    pred_raw: float,
    pi_lo: float,
    pi_hi: float,
    naive_close: float,
):
    outp = out_pred_path(ver, tf_name, pid)
    ts_iso = pd.to_datetime(ts, utc=True).isoformat()

    last_ts = _read_last_time_csv(outp)
    if last_ts is not None and last_ts == ts_iso:
        return

    row = pd.DataFrame([
        {
            "time": ts_iso,
            "last_close": float(last_close),
            "pred_next_close": float(pred_close),
            "pred_next_close_raw": float(pred_raw),
            "pi95_lower": float(pi_lo),
            "pi95_upper": float(pi_hi),
            "naive_next_close": float(naive_close),
        }
    ])

    if os.path.exists(outp):
        try:
            head = pd.read_csv(outp, nrows=1)
            if list(head.columns) != CSV_COLUMNS:
                os.rename(outp, outp.replace(".csv", f"_old_{int(time.time())}.csv"))
        except Exception:
            pass

    if os.path.exists(outp):
        row.to_csv(outp, mode="a", header=False, index=False)
    else:
        row.to_csv(outp, mode="w", header=True, index=False)


# =========================
# Time alignment
# =========================
def align_to_next_15m():
    now = now_utc()
    next_epoch = ((int(now.timestamp()) // BASE_ALIGN_SECONDS) + 1) * BASE_ALIGN_SECONDS
    nxt = datetime.fromtimestamp(next_epoch + 5, tz=timezone.utc)  # +5 seconds buffer
    time.sleep(max(0.0, (nxt - now).total_seconds()))


# =========================
# Data load/bootstrap
# =========================
def load_or_bootstrap(tf_name: str, pid: str) -> pd.DataFrame:
    path = raw_path(tf_name, pid)
    if os.path.exists(path):
        df = pd.read_parquet(path)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
        window = TIMEFRAMES[tf_name]["window"]
        if len(df) > window:
            df = df.iloc[-window:].reset_index(drop=True)
        return df

    df = bootstrap_last_n_public(
        pid,
        n=TIMEFRAMES[tf_name]["window"],
        granularity=TIMEFRAMES[tf_name]["granularity"],
    )
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    df.to_parquet(path, index=False)
    return df


def save_buffer(tf_name: str, pid: str, df: pd.DataFrame):
    df.to_parquet(raw_path(tf_name, pid), index=False)


# =========================
# Continuity fix
# =========================
def ensure_continuity(tf_name: str, pid: str, df_buf: pd.DataFrame) -> pd.DataFrame:
    if df_buf is None or len(df_buf) < 10:
        return df_buf

    step = TIMEFRAMES[tf_name]["seconds"]
    gap_threshold = int(step * 1.2)

    df_buf = df_buf.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    tail = df_buf["time"].iloc[-10:].reset_index(drop=True)
    dt = tail.diff().dt.total_seconds().fillna(step)

    if (dt > gap_threshold).any():
        backfill = bootstrap_last_n_public(
            pid,
            n=200,
            granularity=TIMEFRAMES[tf_name]["granularity"],
        )
        backfill["time"] = pd.to_datetime(backfill["time"], utc=True)
        df_buf = merge_roll(df_buf, backfill, keep_last=TIMEFRAMES[tf_name]["window"])
        df_buf = df_buf.sort_values("time").drop_duplicates("time").reset_index(drop=True)

    return df_buf


# =========================
# LIVE ENGINE
# =========================
def run_live():
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dirs()

    print(
        f"[INFO] Device={device} | Assets={PRODUCT_IDS} | "
        f"Timeframes={list(TIMEFRAMES.keys())} | Versions={list(CONFIGS.keys())} | DisplayTZ=Europe/London | InternalTZ=UTC"
    )

    buffers = {tf_name: {pid: None for pid in PRODUCT_IDS} for tf_name in TIMEFRAMES}
    bundles = {tf_name: {pid: {} for pid in PRODUCT_IDS} for tf_name in TIMEFRAMES}
    last_daily_retrain = {
        tf_name: {pid: now_utc() for pid in PRODUCT_IDS}
        for tf_name in TIMEFRAMES
    }
    pred_counter = {
        tf_name: {pid: {ver: 0 for ver in CONFIGS} for pid in PRODUCT_IDS}
        for tf_name in TIMEFRAMES
    }
    last_drift_retrain = {
        tf_name: {
            pid: {ver: datetime(1970, 1, 1, tzinfo=timezone.utc) for ver in CONFIGS}
            for pid in PRODUCT_IDS
        }
        for tf_name in TIMEFRAMES
    }

    # Startup
    for tf_name in TIMEFRAMES:
        for pid in PRODUCT_IDS:
            df_buf = load_or_bootstrap(tf_name, pid)
            df_buf = ensure_continuity(tf_name, pid, df_buf)
            buffers[tf_name][pid] = df_buf
            print(f"[INFO] {tf_name} | {pid} buffer candles = {len(df_buf)}")

            for ver in VERSIONS_ORDER:
                cfg = CONFIGS[ver]
                bundle = load_bundle(
                    ckpt_dir(ver, tf_name, pid),
                    device=device,
                    is_multi=IS_MULTI,
                    cfg=cfg,
                )
                if bundle is None:
                    print(f"[INFO] [{tf_name}][{pid}] No checkpoint for {ver}. Training...")
                    bundle = train_batch(df_buf, device=device, is_multi=IS_MULTI, cfg=cfg)
                    save_bundle(
                        bundle,
                        ckpt_dir(ver, tf_name, pid),
                        meta={
                            "asset": pid,
                            "timeframe": tf_name,
                            "version": ver,
                            "window": TIMEFRAMES[tf_name]["window"],
                            "is_multi": IS_MULTI,
                            "timestamp_utc": str(now_utc()),
                            "mode": "startup_train",
                            "cfg": cfg.__dict__,
                        },
                    )
                else:
                    print(f"[INFO] [{tf_name}][{pid}] Loaded checkpoint for {ver}.")
                bundles[tf_name][pid][ver] = bundle
                append_log(ver, tf_name, pid, f"{now_utc()} STARTUP ready | candles={len(df_buf)}")

    # Loop
    while True:
        align_to_next_15m()

        for tf_name in TIMEFRAMES:
            granularity = TIMEFRAMES[tf_name]["granularity"]
            window = TIMEFRAMES[tf_name]["window"]

            for pid in PRODUCT_IDS:
                try:
                    prev_last_time = buffers[tf_name][pid]["time"].iloc[-1] if not buffers[tf_name][pid].empty else None

                    latest = fetch_latest_public(
                        pid,
                        limit=200,
                        granularity=granularity,
                    )
                    if latest.empty:
                        for ver in CONFIGS:
                            append_log(ver, tf_name, pid, f"{now_utc()} WARN no latest candles")
                        continue

                    latest["time"] = pd.to_datetime(latest["time"], utc=True)
                    buffers[tf_name][pid] = merge_roll(buffers[tf_name][pid], latest, keep_last=window)
                    buffers[tf_name][pid] = ensure_continuity(tf_name, pid, buffers[tf_name][pid])
                    save_buffer(tf_name, pid, buffers[tf_name][pid])

                    if prev_last_time is None:
                        new_idx = list(range(len(buffers[tf_name][pid])))
                    else:
                        new_rows = buffers[tf_name][pid][buffers[tf_name][pid]["time"] > prev_last_time]
                        if new_rows.empty:
                            continue
                        new_idx = new_rows.index.tolist()

                    for idx in new_idx:
                        df_slice = buffers[tf_name][pid].iloc[: idx + 1].copy().reset_index(drop=True)
                        last_time = df_slice["time"].iloc[-1]
                        last_close = float(df_slice["close"].iloc[-1])
                        naive_pred = last_close

                        # label step (t-1 -> t)
                        if len(df_slice) >= 2:
                            prev_time_iso = pd.to_datetime(df_slice["time"].iloc[-2], utc=True).isoformat()
                            prev_close = float(df_slice["close"].iloc[-2])
                            actual_lr = float(np.log((last_close + 1e-12) / (prev_close + 1e-12)))

                            for ver in VERSIONS_ORDER:
                                bundle = bundles[tf_name][pid][ver]
                                cfg = CONFIGS[ver]

                                update_calibration(bundle, prev_time_iso, actual_lr)

                                if (
                                    bundle.last_pred_time == prev_time_iso
                                    and bundle.last_pred_lr_used is not None
                                    and bundle.last_pred_next_close is not None
                                ):
                                    resid_ret = float(actual_lr - bundle.last_pred_lr_used)
                                    bundle.resid_stats.update(resid_ret)

                                    drift = bundle.ph.update(resid_ret)
                                    if drift:
                                        bundle.drift_flag = True
                                        bundle.last_drift_time = prev_time_iso

                                    err_model = abs(last_close - float(bundle.last_pred_next_close))
                                    err_naive = abs(last_close - prev_close)

                                    q90 = bundle.resid_stats.q_abs(0.90)
                                    outlier = abs(resid_ret) > max(q90, 1e-6)
                                    worse_than_naive = err_model > err_naive

                                    do_update = worse_than_naive or outlier or drift
                                    if do_update:
                                        online_update(bundle, df_slice)

                                    if DRIFT_RETRAIN_ENABLED and drift:
                                        now = now_utc()
                                        cooldown_ok = (
                                            now - last_drift_retrain[tf_name][pid][ver]
                                        ).total_seconds() >= DRIFT_RETRAIN_COOLDOWN_HOURS * 3600
                                        if cooldown_ok:
                                            append_log(ver, tf_name, pid, f"{now} DRIFT detected -> light retrain ({ver})")
                                            bundle_new = train_batch(
                                                df_slice,
                                                device=device,
                                                is_multi=IS_MULTI,
                                                cfg=cfg,
                                                epochs=cfg.daily_epochs,
                                            )
                                            bundles[tf_name][pid][ver] = bundle_new
                                            save_bundle(
                                                bundle_new,
                                                ckpt_dir(ver, tf_name, pid),
                                                meta={
                                                    "asset": pid,
                                                    "timeframe": tf_name,
                                                    "version": ver,
                                                    "timestamp_utc": str(now),
                                                    "mode": "drift_light_retrain",
                                                    "cfg": cfg.__dict__,
                                                },
                                            )
                                            last_drift_retrain[tf_name][pid][ver] = now

                        # predict next (t -> t+1)
                        for ver in VERSIONS_ORDER:
                            cfg = CONFIGS[ver]
                            bundle = bundles[tf_name][pid][ver]

                            pred_close, pred_raw, pi_lo, pi_hi = predict_next(bundle, df_slice)

                            line = (
                                f"engine_time_local={now_local().strftime('%Y-%m-%d %H:%M:%S %Z')} "
                                f"engine_time_utc={now_utc().isoformat()} PRED_{ver} "
                                f"tf={tf_name} candle_time_local={fmt_local(last_time)} "
                                f"candle_time_utc={pd.to_datetime(last_time, utc=True).isoformat()} "
                                f"last_close={last_close:.2f} pred_next_close={pred_close:.2f} pred_raw={pred_raw:.2f} "
                                f"PI95=[{pi_lo:.2f},{pi_hi:.2f}] naive_next_close={naive_pred:.2f}"
                            )
                            print(f"[{tf_name}][{pid}] {line}")
                            append_log(ver, tf_name, pid, line)
                            append_prediction(
                                ver,
                                tf_name,
                                pid,
                                last_time,
                                last_close,
                                pred_close,
                                pred_raw,
                                pi_lo,
                                pi_hi,
                                naive_pred,
                            )

                            pred_counter[tf_name][pid][ver] += 1
                            if pred_counter[tf_name][pid][ver] % SAVE_EVERY_N_PRED == 0:
                                save_bundle(
                                    bundle,
                                    ckpt_dir(ver, tf_name, pid),
                                    meta={
                                        "asset": pid,
                                        "timeframe": tf_name,
                                        "version": ver,
                                        "timestamp_utc": str(now_utc()),
                                        "mode": "periodic_save",
                                        "cfg": cfg.__dict__,
                                    },
                                )

                    # daily retrain
                    now = now_utc()
                    if (now - last_daily_retrain[tf_name][pid]).total_seconds() >= DAILY_RETRAIN_HOURS * 3600:
                        df_buf = bootstrap_last_n_public(
                            pid,
                            n=window,
                            granularity=granularity,
                        )
                        df_buf["time"] = pd.to_datetime(df_buf["time"], utc=True)
                        df_buf = df_buf.sort_values("time").drop_duplicates("time").reset_index(drop=True)
                        df_buf = ensure_continuity(tf_name, pid, df_buf)
                        buffers[tf_name][pid] = df_buf
                        save_buffer(tf_name, pid, buffers[tf_name][pid])

                        for ver in VERSIONS_ORDER:
                            cfg = CONFIGS[ver]
                            bundle = train_batch(df_buf, device=device, is_multi=IS_MULTI, cfg=cfg, epochs=cfg.daily_epochs)
                            bundles[tf_name][pid][ver] = bundle
                            save_bundle(
                                bundle,
                                ckpt_dir(ver, tf_name, pid),
                                meta={
                                    "asset": pid,
                                    "timeframe": tf_name,
                                    "version": ver,
                                    "timestamp_utc": str(now),
                                    "mode": "daily_retrain",
                                    "cfg": cfg.__dict__,
                                },
                            )
                            append_log(ver, tf_name, pid, f"{now} DAILY_RETRAIN done | candles={len(df_buf)}")

                        last_daily_retrain[tf_name][pid] = now

                except Exception as e:
                    err = f"{now_utc()} ERROR {repr(e)}"
                    print(f"[{tf_name}][{pid}] {err}")
                    for ver in CONFIGS:
                        append_log(ver, tf_name, pid, err)


# =========================
# DASHBOARD
# =========================
def run_dashboard():
    import plotly.graph_objects as go
    import streamlit as st
    from streamlit_autorefresh import st_autorefresh

    st.set_page_config(page_title="Crypto Forecasting Dashboard (15m / 30m / 1h)", layout="wide")
    st.title("Live Crypto Forecasting Dashboard")
    st.caption("Timeframes: 15m, 30m, 1h | Models: V1, V2, V3, V3A, V3B | Assets: BTC, ETH, SOL")
    st.caption("Internal model timing/storage stays in UTC. Dashboard timestamps are shown in Europe/London.")

    st_autorefresh(interval=1000, key="refresh_1s")

    def load_raw(tf_name: str, pid: str):
        path = raw_path(tf_name, pid)
        if not os.path.exists(path):
            return None
        df = pd.read_parquet(path)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
        df["time_local"] = df["time"].dt.tz_convert(DISPLAY_TZ)
        return df

    def load_pred(ver: str, tf_name: str, pid: str):
        path = out_pred_path(ver, tf_name, pid)
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        for col in ["pred_next_close_raw", "pi95_lower", "pi95_upper"]:
            if col not in df.columns:
                df[col] = np.nan
        df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
        df["time_local"] = df["time"].dt.tz_convert(DISPLAY_TZ)
        return df

    def safe_corr(a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 3:
            return float("nan")
        a = a[mask]
        b = b[mask]
        sa = float(np.std(a, ddof=1))
        sb = float(np.std(b, ddof=1))
        if sa <= 1e-12 or sb <= 1e-12:
            return float("nan")
        ac = a - float(np.mean(a))
        bc = b - float(np.mean(b))
        return float(np.sum(ac * bc) / ((len(a) - 1) * sa * sb))

    def compute_metrics(df_join: pd.DataFrame, pred_col: str):
        valid = df_join.dropna(subset=["actual_next_close"]).copy()
        if len(valid) < 20:
            return None

        y_true = valid["actual_next_close"].values.astype(float)
        y_pred = valid[pred_col].values.astype(float)
        last_close = valid["last_close"].values.astype(float)

        err = y_true - y_pred
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mape = float(np.mean(np.abs(err) / (y_true + 1e-12)))

        r_act = np.log((y_true + 1e-12) / (last_close + 1e-12))
        r_pred = np.log((y_pred + 1e-12) / (last_close + 1e-12))
        dir_acc = float(np.mean(np.sign(r_act) == np.sign(r_pred)))
        corr = safe_corr(r_act, r_pred)

        return mae, rmse, mape, dir_acc, corr

    tabs = st.tabs(list(TIMEFRAMES.keys()))

    for tab, tf_name in zip(tabs, TIMEFRAMES.keys()):
        with tab:
            st.subheader(f"Timeframe: {tf_name}")

            raw_data = {pid: load_raw(tf_name, pid) for pid in PRODUCT_IDS}
            pred_data = {
                pid: {ver: load_pred(ver, tf_name, pid) for ver in VERSIONS_ORDER}
                for pid in PRODUCT_IDS
            }

            last_n = 200
            pred_points = 2000

            # KPI row
            st.markdown("### Latest Snapshot")
            cols = st.columns(3)
            for i, pid in enumerate(PRODUCT_IDS):
                with cols[i]:
                    st.markdown(f"#### {pid}")
                    df_raw = raw_data.get(pid)
                    if df_raw is None or df_raw.empty:
                        st.warning("No raw data yet. Run engine (`python app.py`).")
                        continue
                    st.write(f"Last time (UK): {df_raw['time_local'].iloc[-1]}")
                    st.caption(f"Last time (UTC): {df_raw['time'].iloc[-1]}")
                    st.metric("Actual close", f"{float(df_raw['close'].iloc[-1]):.2f}")
                    for ver in VERSIONS_ORDER:
                        dfp = pred_data[pid].get(ver)
                        if dfp is not None and not dfp.empty:
                            st.metric(f"{ver.upper()} next close", f"{float(dfp['pred_next_close'].iloc[-1]):.2f}")
                        else:
                            st.caption(f"{ver.upper()} no predictions yet.")

            # OHLC row
            st.markdown(f"### OHLC Candlesticks (last {last_n})")
            cols_ohlc = st.columns(3)
            for i, pid in enumerate(PRODUCT_IDS):
                with cols_ohlc[i]:
                    st.markdown(f"#### {pid}")
                    df_raw = raw_data.get(pid)
                    if df_raw is None or df_raw.empty:
                        st.warning("No raw data yet.")
                        continue
                    df_view = df_raw.iloc[-last_n:].copy()
                    fig = go.Figure(
                        data=[
                            go.Candlestick(
                                x=df_view["time_local"],
                                open=df_view["open"],
                                high=df_view["high"],
                                low=df_view["low"],
                                close=df_view["close"],
                                name="OHLC",
                            )
                        ]
                    )
                    fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10), xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)

            # Prediction row
            st.markdown(f"### Predicted vs Actual (last {pred_points}) — {tf_name}")
            cols_pred = st.columns(3)

            for i, pid in enumerate(PRODUCT_IDS):
                with cols_pred[i]:
                    st.markdown(f"#### {pid}")

                    df_raw = raw_data.get(pid)
                    if df_raw is None or df_raw.empty:
                        st.warning("No raw data yet.")
                        continue

                    df_actual = df_raw[["time", "close"]].copy().sort_values("time")
                    df_actual.rename(columns={"close": "actual_close"}, inplace=True)
                    df_actual["actual_next_close"] = df_actual["actual_close"].shift(-1)

                    for ver in VERSIONS_ORDER:
                        dfp = pred_data[pid].get(ver)
                        if dfp is None or dfp.empty:
                            st.caption(f"{ver.upper()} no predictions yet.")
                            continue

                        dfp = dfp.iloc[-pred_points:].copy()
                        df_join = pd.merge(dfp, df_actual, on="time", how="left")

                        fig2 = go.Figure()
                        x_display = df_join["time"].dt.tz_convert(DISPLAY_TZ)
                        fig2.add_trace(go.Scatter(x=x_display, y=df_join["actual_next_close"], mode="lines", name="Actual next close"))
                        fig2.add_trace(go.Scatter(x=x_display, y=df_join["pred_next_close"], mode="lines", name=f"{ver.upper()} pred"))
                        fig2.add_trace(go.Scatter(x=x_display, y=df_join["naive_next_close"], mode="lines", name="Naive"))

                        if df_join["pi95_lower"].notna().any() and df_join["pi95_upper"].notna().any():
                            fig2.add_trace(go.Scatter(x=x_display, y=df_join["pi95_upper"], mode="lines", name="PI95% upper"))
                            fig2.add_trace(go.Scatter(x=x_display, y=df_join["pi95_lower"], mode="lines", name="PI95% lower"))

                        fig2.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10))
                        st.markdown(f"**{ver.upper()}**")
                        st.plotly_chart(fig2, use_container_width=True)

                        m_model = compute_metrics(df_join, "pred_next_close")
                        m_naive = compute_metrics(df_join, "naive_next_close")
                        if m_model and m_naive:
                            st.write(
                                f"**{ver.upper()} Model** → MAE: {m_model[0]:.3f} | RMSE: {m_model[1]:.3f} | MAPE: {m_model[2]:.4f}\n\n"
                                f"**Naive** → MAE: {m_naive[0]:.3f} | RMSE: {m_naive[1]:.3f} | MAPE: {m_naive[2]:.4f}\n\n"
                                f"DirAcc: {m_model[3]:.3f} | Corr(r): {m_model[4]:.3f}"
                            )
                        else:
                            st.caption("Metrics need ~20+ aligned points.")


# =========================
# Entrypoint
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dashboard", action="store_true", help="Run Streamlit dashboard mode")
    args, _ = parser.parse_known_args()

    if args.dashboard:
        run_dashboard()
    else:
        run_live()


if __name__ == "__main__":
    main()
