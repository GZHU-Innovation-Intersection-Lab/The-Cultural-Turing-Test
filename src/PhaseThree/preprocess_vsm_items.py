# -*- coding: utf-8 -*-
"""
Preprocess VSM 2013 raw responses to long-format item table.

- Reads Excel files: CN-res(model).xlsx and HK-res(model).xlsx
- Note: the input file may contain more than 24 columns (e.g., metadata columns). This script attempts to detect
  and extract the true 24 Likert-item columns automatically.
- Cleans Likert responses with robust parsing (strings, full-width digits, labeled formats, rounding; keeps {1..5} only).
- Produces llm_vsm_items_long.csv with columns: [model, region, persona, item_id, response]
- Writes preprocess_meta.json including official Hofstede scores + ΔH (Mainland − Hong Kong)

Author: Your Name
Date: 2025-09-11
"""

import os
import re
import json
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

# ------------------------------
# Configuration
# ------------------------------
VALID_LIKERT_SET = {1, 2, 3, 4, 5}
LIKERT_ATOL = 1e-6

# Official Hofstede scores (for ΔH reference; direction fixed: Mainland − Hong Kong)
REGION_MAINLAND = "Mainland"
REGION_HONG_KONG = "Hong Kong"
HOFSTEDE_SCORES = {
    REGION_MAINLAND: {'PDI': 80, 'IDV': 43, 'MAS': 66, 'UAI': 30, 'LTO': 77, 'IVR': 24},
    REGION_HONG_KONG: {'PDI': 68, 'IDV': 50, 'MAS': 57, 'UAI': 29, 'LTO': 93, 'IVR': 17}
}
DIMENSIONS = ['PDI', 'IDV', 'MAS', 'UAI', 'LTO', 'IVR']


# ------------------------------
# Robust parsing helpers
# ------------------------------
def _normalize_cell_to_likert(x) -> float:
    """
    Parse a cell into a Likert value in 1..5; returns np.nan if parsing fails.
    Rules:
      - Numeric: allow decimals, round to nearest integer, keep only in [1,5]
      - String: trim, convert full-width digits to half-width, replace common CJK numerals,
        then extract a standalone 1..5; as fallback extract a 0..5 decimal and round
    """
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        v = float(x)
        v_round = float(np.rint(v))
        return v_round if 1.0 <= v_round <= 5.0 else np.nan
    s = str(x).strip()
    if not s:
        return np.nan
    s = s.translate(str.maketrans('０１２３４５６７８９', '0123456789'))
    cjk_map = {
        "\u96f6": "0",
        "\u3007": "0",
        "\u4e00": "1",
        "\u4e8c": "2",
        "\u4e24": "2",
        "\u4e09": "3",
        "\u56db": "4",
        "\u4e94": "5",
    }
    for k, v in cjk_map.items():
        s = s.replace(k, v)
    m = re.search(r'(?<!\d)([1-5])(?!\d)', s)
    if m:
        return float(m.group(1))
    m2 = re.search(r'([0-5](?:\.\d+)?)', s)
    if m2:
        v = float(m2.group(1))
        v_round = float(np.rint(v))
        return v_round if 1.0 <= v_round <= 5.0 else np.nan
    return np.nan


def clean_likert_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robust cleaning: parse values into 1..5 (float). Unparseable values become NaN.
    """
    stacked = df.stack(dropna=False)
    parsed = stacked.map(_normalize_cell_to_likert).astype('float64')
    valid_mask = parsed.isin([1.0, 2.0, 3.0, 4.0, 5.0])
    cleaned = parsed.where(valid_mask, np.nan)
    return cleaned.unstack()


# ------------------------------
# Auto-select 24 Likert columns when raw has >24 columns
# ------------------------------
def auto_select_24_likert_columns(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, List[int]]:
    """
    Auto-select 24 columns that most likely correspond to Likert items, based on
    the per-column parsable ratio into 1..5. Keeps original column order.
    Returns: (df_24cols, chosen_indices)
    """
    ncols = df_raw.shape[1]
    if ncols == 24:
        return df_raw.copy(), list(range(24))
    if ncols < 24:
        raise ValueError(f"Raw column count {ncols} is < 24; cannot form VSM items. Check file format.")

    valid_ratios = []
    for j in range(ncols):
        col = df_raw.iloc[:, j]
        parsed = pd.to_numeric(col, errors='coerce')
        num_valid = parsed.notna().sum()
        if num_valid == 0:
            parsed2 = col.map(_normalize_cell_to_likert)
            num_valid = parsed2.notna().sum()
            ratio = num_valid / max(len(col), 1)
        else:
            ratio = num_valid / max(len(col), 1)
        valid_ratios.append(ratio)

    ratios = np.array(valid_ratios)
    # Heuristic: columns with ratio >= 0.6 are likely item columns; if fewer than 24, take top-24 by ratio.
    candidate_idx = np.where(ratios >= 0.6)[0].tolist()
    if len(candidate_idx) >= 24:
        chosen = sorted(candidate_idx)[-24:]
    else:
        order = np.argsort(ratios)
        chosen = sorted(order[-24:])

    df_24 = df_raw.iloc[:, chosen].copy()
    print(f"Auto-selected 24 columns (0-based indices): {chosen}")
    print(f"Valid ratios (first 10 of chosen): {[round(float(r),3) for r in ratios[chosen][:10]]}")
    return df_24, chosen


# ------------------------------
# Convert cleaned wide (24 columns) to long Q01..Q24
# ------------------------------
def to_long_items(df_clean: pd.DataFrame, model_name: str, region: str, region_code: str) -> pd.DataFrame:
    """
    df_clean: rows = personas, cols = 24 selected item columns, values in {1..5} or NaN
    Returns long-format DataFrame: [model, region, persona, item_id, response]
    """
    wide = df_clean.copy()
    # Temporary integer persona id per file
    wide['persona_tmp'] = np.arange(len(wide), dtype=int)
    # Safe persona ID uses sanitized model name + region code
    safe_model = re.sub(r'[^a-zA-Z0-9_-]+', '-', str(model_name)).strip('-').lower()
    wide['persona'] = [f"{region_code.lower()}-{safe_model}-{i}" for i in wide['persona_tmp']]
    col_map = {i: f"Q{(i+1):02d}" for i in range(24)}
    value_cols = [c for c in wide.columns if c not in ['persona_tmp', 'persona']]
    wide = wide.rename(columns={old: idx for idx, old in enumerate(value_cols)})
    wide_renamed = wide.rename(columns=col_map)

    id_vars = ['persona']
    value_vars = [f"Q{(i+1):02d}" for i in range(24)]
    long_df = wide_renamed.melt(id_vars=id_vars, value_vars=value_vars,
                                var_name='item_id', value_name='response')
    long_df['model'] = model_name
    long_df['region'] = region
    long_df = long_df[['model', 'region', 'persona', 'item_id', 'response']]
    return long_df


# ------------------------------
# Discover and process files
# ------------------------------
def find_and_process_files(directory: str = '.') -> pd.DataFrame:
    """
    Looks for files matching:
      - CN-res(model).xlsx  -> region='Mainland'
      - HK-res(model).xlsx  -> region='Hong Kong'
    Returns concatenated long-format DataFrame from all files.
    """
    all_scores: List[pd.DataFrame] = []
    file_pattern = re.compile(r'^(cn|hk)-res\(([^)]+)\)\.xlsx$', re.IGNORECASE)

    print("\n--- Searching and processing raw data files ---")
    files_found = 0
    xls_unsupported = []

    for filename in sorted(os.listdir(directory)):
        fname = filename.strip()
        if fname.lower().endswith('.xls'):
            xls_unsupported.append(fname)
            continue
        m = file_pattern.match(fname)
        if not m:
            continue

        files_found += 1
        region_code, model_name = m.groups()
        region = REGION_MAINLAND if region_code.lower() == 'cn' else REGION_HONG_KONG
        path = os.path.join(directory, fname)
        print(f"  > Processing: {fname}  (model: {model_name}, region: {region})")

        try:
            df_raw = pd.read_excel(path, header=None, engine='openpyxl')
            if df_raw.shape[1] < 24:
                raise ValueError(f"{fname} has {df_raw.shape[1]} columns, fewer than 24.")
            df_24, chosen = auto_select_24_likert_columns(df_raw)
            df_clean = clean_likert_data(df_24)
            df_long = to_long_items(df_clean, model_name, region, region_code)
            all_scores.append(df_long)
        except Exception as e:
            print(f"    Error processing {fname}: {e}")

    if not all_scores:
        if xls_unsupported:
            raise FileNotFoundError(
                "Detected unsupported .xls files; convert them to .xlsx and retry:\n  - " + "\n  - ".join(xls_unsupported)
            )
        raise FileNotFoundError("No files found matching 'CN|HK-res(model).xlsx'.")

    print(f"--- Successfully processed {files_found} files ---")
    return pd.concat(all_scores, ignore_index=True)


# ------------------------------
# Main
# ------------------------------
def main():
    print("--- VSM 2013 Preprocess (produce 24-item long table) ---")
    try:
        items_long = find_and_process_files()
        before = len(items_long)
        items_long = items_long.dropna(subset=['response']).copy()
        after = len(items_long)
        dropped = before - after

        out_csv = 'llm_vsm_items_long.csv'
        items_long.to_csv(out_csv, index=False)
        print(f"\nWrote: {out_csv}  (kept {after}, dropped {dropped})")

        delta_h = {dim: HOFSTEDE_SCORES[REGION_MAINLAND][dim] - HOFSTEDE_SCORES[REGION_HONG_KONG][dim] for dim in DIMENSIONS}
        files = sorted([f for f in os.listdir('.') if re.match(r'^(cn|hk)-res\([^)]+\)\.xlsx$', f, re.IGNORECASE)])
        meta = {
            "valid_likert_set": sorted(list(VALID_LIKERT_SET)),
            "likert_atol": LIKERT_ATOL,
            "files_processed": files,
            "hofstede_scores_official": HOFSTEDE_SCORES,
            "delta_h_official": delta_h,
            "dimensions_order": DIMENSIONS,
            "note": "Direction for Δ is fixed as Mainland − Hong Kong."
        }
        with open('preprocess_meta.json', 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=True, indent=2)
        print("Wrote: preprocess_meta.json (includes official scores and ΔH)")

        print("\nNext: run analysis_vsm_indices.py to compute VSM indices and ΔL.")

    except Exception as e:
        print(f"\nPreprocess failed: {e}")
