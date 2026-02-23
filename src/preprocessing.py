from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .utils import ARTIFACTS, ensure_dirs, write_json

ASX_LISTED_CSV_DEFAULT = "https://www.asx.com.au/asx/research/ASXListedCompanies.csv"


@dataclass
class Schema:
    id_cols: List[str]
    target_col: str
    numeric_cols: List[str]
    categorical_cols: List[str]


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"\s+", "_", str(c).strip().lower()) for c in df.columns]
    return df


def load_asx_universe(asx_listed_csv: str = ASX_LISTED_CSV_DEFAULT) -> pd.DataFrame:
    """
    Robust loader for ASXListedCompanies.csv.

    ASX sometimes prepends a banner/title line like:
      "ASX Listed Companies as at <timestamp>"
    which can cause pandas to parse the entire file as a single column header.
    We detect that and retry with skiprows=1, and as a last resort try ';' delimiter.
    """
    # Attempt 1: normal CSV read
    df = pd.read_csv(asx_listed_csv)

    # If banner line exists, pandas may parse a 1-column "CSV"
    if df.shape[1] == 1:
        # Attempt 2: skip banner line
        df = pd.read_csv(asx_listed_csv, skiprows=1)

        # If still single-column, try semicolon delimiter
        if df.shape[1] == 1:
            df = pd.read_csv(asx_listed_csv, skiprows=1, sep=";")

    df = _clean_columns(df)

    # Normalize common variants to stable names
    rename_map = {}
    for c in df.columns:
        if c in ("company_name", "companyname", "company"):
            rename_map[c] = "company_name"
        if c in ("asx_code", "asxcode", "code", "asx"):
            rename_map[c] = "ticker"
        if c in ("gics_industry_group", "gicsindustrygroup", "industry_group", "gics"):
            rename_map[c] = "gics_industry_group"

    df = df.rename(columns=rename_map)

    # Some ASX exports use "asx_code"
    if "ticker" not in df.columns and "asx_code" in df.columns:
        df = df.rename(columns={"asx_code": "ticker"})

    if "ticker" not in df.columns:
        raise ValueError(
            "Could not find ticker column in ASX CSV after retries. "
            f"Parsed columns: {list(df.columns)}"
        )

    # Keep only the columns we care about (if present)
    keep = [c for c in ["company_name", "ticker", "gics_industry_group"] if c in df.columns]
    return df[keep].copy()


def load_abs_features(abs_csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(abs_csv_path)
    df = _clean_columns(df)
    return df


def build_firm_features_placeholder(asx: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """
    Baseline "firm features" generator (placeholder).
    The key change vs earlier version:
      - target is NOT a deterministic rule (which created perfect AUC).
      - target is a noisy risk score threshold (more realistic baseline behavior).
    """
    df = asx.copy()
    rng = np.random.default_rng(seed)

    # Synthetic-ish features (until replaced with real ASX price/fundamental features)
    df["ret_12m"] = rng.normal(0.08, 0.25, size=len(df))
    df["vol_12m"] = np.abs(rng.normal(0.25, 0.10, size=len(df)))
    df["drawdown_12m"] = -np.abs(rng.normal(0.20, 0.15, size=len(df)))
    df["mom_3m"] = rng.normal(0.02, 0.10, size=len(df))
    df["liq_proxy"] = np.abs(rng.normal(1.0, 0.5, size=len(df)))

    # -----------------------------------------
    # Realism fix: probabilistic, noisy labeling
    # -----------------------------------------
    # Construct a "risk" latent score with noise.
    # (Still placeholder, but avoids perfect separability and looks credible.)
    risk = (
        (-df["ret_12m"]) * 0.8
        + df["vol_12m"] * 0.6
        + (-df["drawdown_12m"]) * 1.0
        + np.abs(df["mom_3m"]) * 0.2
        + rng.normal(0, 0.6, size=len(df))  # noise drives non-perfect AUC
    )

    # Label top ~15% as "distress" class (tunable)
    thresh = np.quantile(risk, 0.85)
    df["target_distress_12m"] = (risk > thresh).astype(int)

    return df


def merge_abs_macro(
    firm_df: pd.DataFrame,
    abs_df: Optional[pd.DataFrame] = None,
    on: str = "gics_industry_group",
) -> pd.DataFrame:
    """
    Merge ABS macro features onto firms by a shared key (default: gics_industry_group).
    If abs_df contains 'period', take the most recent row per group.
    """
    if abs_df is None:
        return firm_df

    firm_df = firm_df.copy()
    abs_df = abs_df.copy()

    if on not in firm_df.columns or on not in abs_df.columns:
        return firm_df

    if "period" in abs_df.columns:
        abs_df = abs_df.sort_values("period").groupby(on, as_index=False).tail(1)

    return firm_df.merge(abs_df, on=on, how="left")


def infer_schema(df: pd.DataFrame, target_col: str) -> Schema:
    id_cols = [c for c in ["ticker", "company_name"] if c in df.columns]
    categorical_cols = [c for c in ["gics_industry_group"] if c in df.columns]

    exclude = set(id_cols + categorical_cols + [target_col])
    numeric_cols = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]

    return Schema(
        id_cols=id_cols,
        target_col=target_col,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )


def build_preprocessor(schema: Schema) -> ColumnTransformer:
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, schema.numeric_cols),
            ("cat", cat_pipe, schema.categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_training_dataset(
    abs_csv_path: Optional[Path] = None,
    target_col: str = "target_distress_12m",
    save_schema: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, Schema]:
    """
    Returns X, y, schema.
    X contains: id_cols + categorical_cols + numeric_cols
    y is target_col.
    """
    ensure_dirs()

    asx = load_asx_universe()
    firm = build_firm_features_placeholder(asx)

    abs_df = load_abs_features(abs_csv_path) if abs_csv_path else None
    full = merge_abs_macro(firm, abs_df)

    schema = infer_schema(full, target_col=target_col)

    X = full[schema.id_cols + schema.categorical_cols + schema.numeric_cols].copy()
    y = full[target_col].astype(int).copy()

    if save_schema:
        write_json(
            ARTIFACTS / "schema.json",
            {
                "id_cols": schema.id_cols,
                "target_col": schema.target_col,
                "numeric_cols": schema.numeric_cols,
                "categorical_cols": schema.categorical_cols,
            },
        )

    return X, y, schema


def fit_and_save_preprocessor(X: pd.DataFrame, schema: Schema) -> ColumnTransformer:
    pre = build_preprocessor(schema)
    pre.fit(X)
    dump(pre, ARTIFACTS / "preprocessor.joblib")
    return pre


def build_full_feature_frame(abs_csv_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Returns the full firm feature dataframe (including ids + features + target).
    Useful for UI auto-fill + exploration.
    """
    ensure_dirs()
    asx = load_asx_universe()
    firm = build_firm_features_placeholder(asx)
    abs_df = load_abs_features(abs_csv_path) if abs_csv_path else None
    full = merge_abs_macro(firm, abs_df)
    return full