# coinbase.py  (PUBLIC candles, robust + correct)
from __future__ import annotations

import time
import requests
import pandas as pd
import numpy as np

BASE_URL = "https://api.coinbase.com/api/v3/brokerage/market/products"

# Coinbase Advanced Trade granularity enums + their seconds
# (critical: FIFTEEN_MINUTE = 900 seconds)
GRANULARITY_SECONDS = {
    "ONE_MINUTE": 60,
    "FIVE_MINUTE": 300,
    "FIFTEEN_MINUTE": 900,
    "THIRTY_MINUTE": 1800,
    "ONE_HOUR": 3600,
    "TWO_HOUR": 7200,
    "FOUR_HOUR": 14400,
    "SIX_HOUR": 21600,
    "ONE_DAY": 86400,
}

MAX_LIMIT = 350  # docs: max 350 :contentReference[oaicite:1]{index=1}

def _align_end_to_bucket(now_ts: int, granularity: str) -> int:
    """Align end to the most recent completed bucket boundary."""
    step = GRANULARITY_SECONDS[granularity]
    return now_ts - (now_ts % step)

def _candles_request(
    product_id: str,
    start_ts: int,
    end_ts: int,
    granularity: str,
    limit: int,
    timeout: int = 20,
    retries: int = 4,
):
    url = f"{BASE_URL}/{product_id}/candles"
    params = {
        "start": str(int(start_ts)),
        "end": str(int(end_ts)),
        "granularity": granularity,
        "limit": int(limit),
    }

    last_err = None
    for k in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)

            # 400 is NOT retryable usually; it means your params are invalid
            if r.status_code == 400:
                raise RuntimeError(f"400 Bad Request. Params={params}. Body={r.text[:200]}")

            r.raise_for_status()
            return r.json()

        except Exception as e:
            last_err = e
            # small exponential backoff
            time.sleep(min(2 ** k, 8))

    raise RuntimeError(f"Request failed after retries. Last error: {repr(last_err)}")

def _json_to_df(js: dict) -> pd.DataFrame:
    candles = js.get("candles", [])
    if not candles:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(candles)
    # expected keys: start, low, high, open, close, volume (strings) :contentReference[oaicite:2]{index=2}
    df = df.rename(columns={"start": "time"})
    df["time"] = pd.to_datetime(df["time"].astype(np.int64), unit="s", utc=True)

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[["time", "open", "high", "low", "close", "volume"]]
    df = df.dropna(subset=["time", "close"]).sort_values("time").drop_duplicates("time").reset_index(drop=True)
    return df

def fetch_latest_public(product_id: str, limit: int = 3, granularity: str = "FIFTEEN_MINUTE") -> pd.DataFrame:
    """
    Fetch latest N candles. Ensures (end-start)/bucket_seconds <= limit to avoid 400.
    """
    limit = int(max(1, min(limit, MAX_LIMIT)))
    if granularity not in GRANULARITY_SECONDS:
        raise ValueError(f"Unknown granularity: {granularity}")

    step = GRANULARITY_SECONDS[granularity]
    end_ts = _align_end_to_bucket(int(time.time()), granularity)
    start_ts = end_ts - (limit * step)

    js = _candles_request(product_id, start_ts, end_ts, granularity, limit)
    df = _json_to_df(js)

    # Sometimes API may return more/less due to boundary behavior; keep last "limit"
    if len(df) > limit:
        df = df.iloc[-limit:].reset_index(drop=True)
    return df

def bootstrap_last_n_public(product_id: str, n: int = 6000, granularity: str = "FIFTEEN_MINUTE") -> pd.DataFrame:
    """
    Fetch last N candles by paging backwards. Each request respects limit<=350 and time range<=limit*bucketsize.
    """
    n = int(max(1, n))
    if granularity not in GRANULARITY_SECONDS:
        raise ValueError(f"Unknown granularity: {granularity}")
    step = GRANULARITY_SECONDS[granularity]

    end_ts = _align_end_to_bucket(int(time.time()), granularity)
    chunks = []
    remaining = n

    # page backwards
    while remaining > 0:
        lim = min(MAX_LIMIT, remaining)
        start_ts = end_ts - (lim * step)

        js = _candles_request(product_id, start_ts, end_ts, granularity, lim)
        df = _json_to_df(js)
        if df.empty:
            break

        chunks.append(df)

        # move window backwards (avoid overlap)
        earliest = int(df["time"].iloc[0].timestamp())
        end_ts = earliest - step
        remaining -= len(df)

        # safety: prevent infinite loop if time doesn't move
        if end_ts <= 0:
            break

    if not chunks:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

    out = pd.concat(chunks, ignore_index=True)
    out = out.sort_values("time").drop_duplicates("time").reset_index(drop=True)

    # keep last N only
    if len(out) > n:
        out = out.iloc[-n:].reset_index(drop=True)
    return out

def merge_roll(buffer_df: pd.DataFrame, new_df: pd.DataFrame, keep_last: int = 6000) -> pd.DataFrame:
    """
    Merge new candles into buffer and keep last N rows.
    """
    if buffer_df is None or buffer_df.empty:
        out = new_df.copy()
    else:
        out = pd.concat([buffer_df, new_df], ignore_index=True)

    out["time"] = pd.to_datetime(out["time"], utc=True)
    out = out.sort_values("time").drop_duplicates("time").reset_index(drop=True)

    if len(out) > keep_last:
        out = out.iloc[-keep_last:].reset_index(drop=True)
    return out