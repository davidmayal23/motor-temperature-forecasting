#!/usr/bin/env python3

# Myscellaneous
import os
import sys
from pathlib import Path

# Data manipulation
import pandas as pd
import numpy as np

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def load_psm_data(file_path: Path, dt_sec: float = 0.5):
    # Load PSM data from a CSV file
    if file_path.exists() or os.path.exists(file_path):
        df = pd.read_csv(file_path, low_memory=False)
    else:
        raise FileNotFoundError("No data file found. Provide a valid CSV path.")
    # Group profiles by ID and create a time column
    df["time"] = df.groupby("profile_id").cumcount() * dt_sec
    # Fix profile_id to be sequential integers
    df["profile_id"] = df.groupby("profile_id").ngroup() + 1
    
    # Reorder columns to have profile_id and time first
    cols = df.columns.tolist()
    cols.insert(0, cols.pop(cols.index("time")))
    cols.insert(0, cols.pop(cols.index("profile_id")))
    df = df[cols]
    
    return df

def build_data_summary(df: pd.DataFrame):
    grouped_df = df.groupby("profile_id")
    mean_duration = grouped_df["time"].max().mean()
    sampling_freq = 1 / (df["time"].diff().median())
    num_profiles = df["profile_id"].nunique()
    mem_mb = round(df.memory_usage(deep=True).sum() / (1024**2), 3)
    missing_pct = round(df.isna().mean().mean() * 100, 2)
    summary = {
        "Number of Rows": len(df),
        "Number of Columns": len(df.columns),
        "Number of Profiles": num_profiles,
        "Mean Duration (s)": round(mean_duration, 2),
        "Sampling Frequency (Hz)": round(sampling_freq, 2),
        "Memory Usage (MB)": mem_mb,
        "Missing Data (%)": missing_pct
    }
    return summary

def profile_variance_summary(df: pd.DataFrame, group_var: str = "profile_id"):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop([group_var])
    stats=[]
    for profile_id, group in df.groupby(group_var):
        for col in numeric_cols:
            series = group[col]
            if len(series) < 2:
                continue
            mean = series.mean()
            std = series.std()
            var = series.var()
            coef_var = std / mean if mean != 0 else np.nan
            stats.append({
                "profile_id": profile_id,
                "feature": col,
                "mean": mean,
                "std": std,
                "var": var,
                "coef_var": coef_var
            })
    return pd.DataFrame(stats)

def _require_persistence(mask: pd.Series, groups: pd.Series, n: int = 2):
    if n <= 1:
        return mask.fillna(False)
    m = mask.copy()
    for k in range(1, n):
        m = m & mask.groupby(groups, sort=False).shift(-k).fillna(False)
    return m.fillna(False)

def check_derivatives(df: pd.DataFrame, checks: dict, time_var: str, require_pers: bool = False, group_var: str = "profile_id"):
    # Keep only what we need
    base = df.sort_values([group_var, time_var], kind="stable")
    grp = base[group_var]
    g = base.groupby(group_var, sort=False)

    # Start output with keys
    out = base[[group_var, time_var]].copy()

    for col, params in checks.items():
        abs_th = float(params["abs"])
        rel_th = float(params["rel"])
        require_pers = bool(params.get("require_pers", False))
        pers_len = int(params.get("persist_len", 2))
        # absolute mask
        diff_abs = g[col].diff().abs()
        mask_abs = (diff_abs + 1e-9) > abs_th
        # relative mask vs previous magnitude
        prev_abs = g[col].shift(1).abs()
        mask_rel = (prev_abs > 1.0) & ((diff_abs / (prev_abs + 1e-9)) > rel_th)
        # union of the masks
        mask_any = mask_abs | mask_rel

        # write columns
        out[f"{col}_diff_abs"] = diff_abs.values
        out[f"{col}_mask_abs"]  = mask_abs.values
        out[f"{col}_mask_rel"]  = mask_rel.values
        out[f"{col}_mask_any"]  = mask_any.values

        # optional persistence (groupwise)
        if require_pers:
            m_persist = _require_persistence(mask_any, groups=grp, n=pers_len)
            out[f"{col}_mask_persist"] = m_persist.values

    return out.reset_index(drop=True)

def extract_profile_features(df, group_var="profile_id", time_var="time"):
    df.sort_values([group_var, time_var], kind="stable", inplace=True)
    feats = []
    for profile_id, g in df.groupby(group_var):
        feat_dict = {"profile_id": profile_id}

        duration = g[time_var].max() - g[time_var].min()
        DT = g["time"].diff().median()
        feat_dict["duration_s"] = duration

        g.drop(columns=[group_var, time_var], inplace=True)
        for col in g.select_dtypes(include=[np.number]).columns:
            s = g[col].dropna()
            if s.empty: 
                continue

            # Basic stats
            feat_dict[f"{col}_mean"] = s.mean()
            feat_dict[f"{col}_median"] = s.median()
            feat_dict[f"{col}_std"] = s.std()
            # feat_dict[f"{col}_var"] = s.var()
            feat_dict[f"{col}_min"] = s.min()
            feat_dict[f"{col}_max"] = s.max()
            feat_dict[f"{col}_range"] = s.max() - s.min()
            
            # Percentiles
            feat_dict[f"{col}_p10"] = np.percentile(s, 10)
            feat_dict[f"{col}_p90"] = np.percentile(s, 90)

            # Slopes / dynamics
            diffs = s.diff().dropna() / DT
            if not diffs.empty:
                feat_dict[f"{col}_max_slope"] = diffs.abs().max()
                feat_dict[f"{col}_median_slope"] = diffs.median()

            # Autocorrelation at lag-1
            if len(s) > 1:
                feat_dict[f"{col}_acf1"] = s.autocorr(lag=1)

        feats.append(feat_dict)

    return pd.DataFrame(feats).set_index("profile_id")

def repair_anomalies(df: pd.DataFrame, mask_df: pd.DataFrame, checks: dict, time_var: str, group_var: str = "profile_id", _prev_flags: int = None):
    df_repaired = df.sort_values([group_var, time_var], kind="stable").copy()
    if mask_df is None:
        mask_df = check_derivatives(df_repaired, checks, time_var=time_var, group_var=group_var)
    
    def _repair_group(g):
        g = g.interpolate(method='linear', limit_area='inside')
        g = g.ffill()
        g = g.bfill()
        return g
    
    cols = list(checks.keys())
    for col in cols:
        mask_col = f"{col}_mask_persist" if f"{col}_mask_persist" in mask_df.columns else f"{col}_mask_any"
        m = mask_df[mask_col].to_numpy()
        if not m.any():
            continue
        # col_pos = df_repaired.columns.get_loc(col)        
        # df_repaired.iloc[m, col_pos] = np.nan
        df_repaired.loc[m, col] = np.nan
        df_repaired[col] = df_repaired.groupby(group_var)[col].transform(_repair_group)
        print(f"Repaired {mask_df[mask_col].sum()} anomalies in column '{col}'")
    
    mask_df = check_derivatives(df_repaired, checks, time_var=time_var, group_var=group_var)
    flags_now = int(mask_df.filter(like='_mask_').to_numpy(dtype=bool).sum())
    nan_mask = df_repaired[cols].isna()
    if _prev_flags is not None and flags_now >= _prev_flags:
        df_repaired[nan_mask] = df[nan_mask]
        return df_repaired
    elif (df_repaired.isna().sum().sum() > 0 or mask_df.filter(like='_mask_').sum().sum() > 0):
        df_repaired = repair_anomalies(df_repaired, mask_df, checks, time_var, group_var, _prev_flags=flags_now)
    
    df_repaired[nan_mask] = df[nan_mask]
    return df_repaired
    

if __name__ == "__main__":
    # Example usage
    data_file = Path("data/raw/measures_v2.csv")
    df = load_psm_data(data_file)
    # print(df.head())
    # print(build_data_summary(df))
    # print(profile_variance_summary(df))
    # print(check_derivatives(df, {"torque": {"abs": 0.5, "rel": 0.1, "require_pers": True, "persist_len": 3}}, time_var="time").head(10))
    checks = {
    "stator_winding": {"abs": 3, "rel": 0.05},
    "stator_tooth":   {"abs": 3, "rel": 0.05},
    "stator_yoke":    {"abs": 3, "rel": 0.05},
    "pm":             {"abs": 3, "rel": 0.05},
    "ambient":        {"abs": 0.5, "rel": 0.05},
    # "torque":         {"abs": 100, "rel": 2.0, "require_pers": True, "pers_len": 4}
    }

    df_repaired = repair_anomalies(
    df,
    None,
    checks,
    time_var="time",
    group_var="profile_id",
    )
