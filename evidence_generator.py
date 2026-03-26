from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_PRED_COLUMNS = [
    "time",
    "target_time",
    "last_close",
    "pred_next_close",
    "pred_next_close_raw",
    "pi95_lower",
    "pi95_upper",
    "naive_next_close",
]


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
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



def compute_metrics(df_join: pd.DataFrame, pred_col: str) -> Optional[dict]:
    valid = df_join.dropna(subset=["actual_next_close", pred_col, "last_close"]).copy()
    if len(valid) < 5:
        return None

    y_true = valid["actual_next_close"].values.astype(float)
    y_pred = valid[pred_col].values.astype(float)
    last_close = valid["last_close"].values.astype(float)

    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mape = float(np.mean(np.abs(err) / (np.abs(y_true) + 1e-12)))

    r_act = np.log((y_true + 1e-12) / (last_close + 1e-12))
    r_pred = np.log((y_pred + 1e-12) / (last_close + 1e-12))
    dir_acc = float(np.mean(np.sign(r_act) == np.sign(r_pred)))
    corr = safe_corr(r_act, r_pred)

    return {
        "rows": int(len(valid)),
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "dir_acc": dir_acc,
        "corr_r": corr,
    }



def load_raw(raw_root: Path, tf_name: str, pid: str) -> Optional[pd.DataFrame]:
    path = raw_root / tf_name / pid / "buffer.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "time" not in df.columns or "close" not in df.columns:
        return None
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values("time").drop_duplicates("time").reset_index(drop=True)



def load_pred(out_root: Path, ver: str, tf_name: str, pid: str) -> Optional[pd.DataFrame]:
    path = out_root / tf_name / pid / f"predictions_{ver}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    for col in DEFAULT_PRED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    if "target_time" in df.columns:
        df["target_time"] = pd.to_datetime(df["target_time"], utc=True, errors="coerce")
    return df.sort_values("time").drop_duplicates("time").reset_index(drop=True)



def join_actuals(df_raw: pd.DataFrame, df_pred: pd.DataFrame) -> pd.DataFrame:
    actual_map = df_raw[["time", "close"]].copy().sort_values("time")
    actual_map.rename(columns={"time": "target_time", "close": "actual_next_close"}, inplace=True)

    if "target_time" in df_pred.columns and df_pred["target_time"].notna().any():
        df_join = pd.merge(df_pred, actual_map, on="target_time", how="left")
    else:
        df_actual = df_raw[["time", "close"]].copy().sort_values("time")
        df_actual.rename(columns={"close": "actual_close"}, inplace=True)
        df_actual["actual_next_close"] = df_actual["actual_close"].shift(-1)
        df_join = pd.merge(df_pred, df_actual[["time", "actual_next_close"]], on="time", how="left")

    return df_join.sort_values("time").reset_index(drop=True)



def discover_versions(base_dir: Path) -> list[str]:
    versions: list[str] = []
    for p in sorted(base_dir.glob("data_out_*")):
        if p.is_dir():
            name = p.name.replace("data_out_", "")
            versions.append(name)
    return versions



def discover_timeframes(raw_root: Path) -> list[str]:
    return sorted([p.name for p in raw_root.iterdir() if p.is_dir()]) if raw_root.exists() else []



def discover_assets(raw_root: Path, tf_name: str) -> list[str]:
    tf_dir = raw_root / tf_name
    return sorted([p.name for p in tf_dir.iterdir() if p.is_dir()]) if tf_dir.exists() else []



def plot_pred_vs_actual(df_join: pd.DataFrame, title: str, out_png: Path, last_n: int) -> None:
    plot_df = df_join.dropna(subset=["actual_next_close", "pred_next_close"]).copy()
    if plot_df.empty:
        return
    plot_df = plot_df.iloc[-last_n:].copy()

    plt.figure(figsize=(12, 6))
    plt.plot(plot_df["time"], plot_df["actual_next_close"], label="Actual next close")
    plt.plot(plot_df["time"], plot_df["pred_next_close"], label="Model prediction")
    if "naive_next_close" in plot_df.columns and plot_df["naive_next_close"].notna().any():
        plt.plot(plot_df["time"], plot_df["naive_next_close"], label="Naive baseline")
    if "pi95_lower" in plot_df.columns and "pi95_upper" in plot_df.columns:
        if plot_df["pi95_lower"].notna().any() and plot_df["pi95_upper"].notna().any():
            plt.fill_between(
                plot_df["time"],
                plot_df["pi95_lower"],
                plot_df["pi95_upper"],
                alpha=0.2,
                label="PI95",
            )

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()



def fmt_metric(x: float) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "nan"
    return f"{x:.6f}"



def write_markdown_summary(summary_path: Path, rows: list[dict]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Evidence of Results",
        "",
        "This file was generated by `evidence_generator.py` from the existing raw buffers and prediction CSV files.",
        "",
        "## Metrics Summary",
        "",
        "| Timeframe | Asset | Version | Rows | Model MAE | Model RMSE | Model MAPE | DirAcc | Corr(r) | Naive MAE | Naive RMSE | Naive MAPE | Plot | Sample |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['timeframe']} | {r['asset']} | {r['version']} | {r['rows']} | {fmt_metric(r['model_mae'])} | {fmt_metric(r['model_rmse'])} | {fmt_metric(r['model_mape'])} | {fmt_metric(r['dir_acc'])} | {fmt_metric(r['corr_r'])} | {fmt_metric(r['naive_mae'])} | {fmt_metric(r['naive_rmse'])} | {fmt_metric(r['naive_mape'])} | {r['plot_file']} | {r['sample_file']} |"
        )
    summary_path.write_text("\n".join(lines), encoding="utf-8")



def main() -> None:
    parser = argparse.ArgumentParser(description="Generate saved plots, metrics, and sample output from existing crypto forecast files.")
    parser.add_argument("--base-dir", default=".", help="Project root that contains data_raw and data_out_* folders.")
    parser.add_argument("--output-dir", default="evidence_output", help="Folder where plots, metrics, and sample outputs will be saved.")
    parser.add_argument("--assets", nargs="*", default=None, help="Optional asset filter, for example BTC-USD ETH-USD")
    parser.add_argument("--timeframes", nargs="*", default=None, help="Optional timeframe filter, for example 15m 30m 1h")
    parser.add_argument("--versions", nargs="*", default=None, help="Optional version filter, for example v1 v2 v3 v3a v3b v4")
    parser.add_argument("--pred-points", type=int, default=500, help="Number of latest aligned points to use in each saved plot.")
    parser.add_argument("--sample-rows", type=int, default=10, help="Number of latest rows to export in each sample output CSV.")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    raw_root = base_dir / "data_raw"
    output_dir = Path(args.output_dir).resolve()
    images_dir = output_dir / "plots"
    samples_dir = output_dir / "sample_outputs"
    metrics_csv = output_dir / "metrics_summary.csv"
    metrics_md = output_dir / "proof_of_results.md"

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    if not raw_root.exists():
        raise SystemExit(f"Could not find raw data folder: {raw_root}")

    timeframes = args.timeframes or discover_timeframes(raw_root)
    if not timeframes:
        raise SystemExit("No timeframe folders were found inside data_raw.")

    discovered_versions = discover_versions(base_dir)
    versions = args.versions or discovered_versions
    if not versions:
        raise SystemExit("No data_out_* folders were found. Run your live script first so predictions exist.")

    all_rows: list[dict] = []
    processed = 0

    for tf_name in timeframes:
        assets = args.assets or discover_assets(raw_root, tf_name)
        for pid in assets:
            df_raw = load_raw(raw_root, tf_name, pid)
            if df_raw is None or df_raw.empty:
                continue

            for ver in versions:
                out_root = base_dir / f"data_out_{ver}"
                df_pred = load_pred(out_root, ver, tf_name, pid)
                if df_pred is None or df_pred.empty:
                    continue

                df_join = join_actuals(df_raw, df_pred)
                model_metrics = compute_metrics(df_join, "pred_next_close")
                naive_metrics = compute_metrics(df_join, "naive_next_close")
                if model_metrics is None:
                    continue

                safe_name = f"{tf_name}__{pid}__{ver}"
                plot_path = images_dir / f"{safe_name}.png"
                sample_path = samples_dir / f"{safe_name}__sample.csv"

                plot_pred_vs_actual(
                    df_join=df_join,
                    title=f"Predicted vs Actual — {tf_name} | {pid} | {ver.upper()}",
                    out_png=plot_path,
                    last_n=args.pred_points,
                )

                sample_cols = [
                    c for c in [
                        "time",
                        "target_time",
                        "last_close",
                        "pred_next_close",
                        "naive_next_close",
                        "actual_next_close",
                        "pi95_lower",
                        "pi95_upper",
                    ] if c in df_join.columns
                ]
                sample_path.parent.mkdir(parents=True, exist_ok=True)
                df_join[sample_cols].tail(args.sample_rows).to_csv(sample_path, index=False)

                all_rows.append(
                    {
                        "timeframe": tf_name,
                        "asset": pid,
                        "version": ver,
                        "rows": model_metrics["rows"],
                        "model_mae": model_metrics["mae"],
                        "model_rmse": model_metrics["rmse"],
                        "model_mape": model_metrics["mape"],
                        "dir_acc": model_metrics["dir_acc"],
                        "corr_r": model_metrics["corr_r"],
                        "naive_mae": None if naive_metrics is None else naive_metrics["mae"],
                        "naive_rmse": None if naive_metrics is None else naive_metrics["rmse"],
                        "naive_mape": None if naive_metrics is None else naive_metrics["mape"],
                        "plot_file": str(plot_path.relative_to(output_dir)),
                        "sample_file": str(sample_path.relative_to(output_dir)),
                    }
                )
                processed += 1

    if not all_rows:
        raise SystemExit(
            "No aligned prediction rows were found. Make sure your live script has already created data_out_* prediction CSV files."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).sort_values(["timeframe", "asset", "version"]).to_csv(metrics_csv, index=False)
    write_markdown_summary(metrics_md, all_rows)

    print(f"Generated evidence for {processed} asset/timeframe/version combinations.")
    print(f"Metrics CSV: {metrics_csv}")
    print(f"Markdown summary: {metrics_md}")
    print(f"Plots folder: {images_dir}")
    print(f"Sample outputs folder: {samples_dir}")


if __name__ == "__main__":
    main()
