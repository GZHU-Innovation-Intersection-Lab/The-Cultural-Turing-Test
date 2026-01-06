# -*- coding: utf-8 -*-
"""
Compute VSM 2013 indices (PDI/IDV/MAS/UAI/LTO/IVR), Delta-L (Mainland − Hong Kong),
and run four-tier validation against official Delta-H with full, ready-to-use statistical tests.
Now with multiple-comparison correction across models per Tier (Holm / BH-FDR).

Inputs:
- llm_vsm_items_long.csv  (columns: model, region, persona, item_id [Q01..Q24], response)
- preprocess_meta.json    (optional; provides official Hofstede scores and ΔH)

Outputs:
- vsm_item_means.csv
- vsm_indices_by_model_region.csv
- deltaL_by_model.json
- deltaH_official.json
- tier_results.json            (includes raw and adjusted p-values)
- tier_results_corrected.json  (redundant copy for convenience)

Author: Your Name
Date: 2025-09-09 (updated with MTC)
"""

import json
import math
import numpy as np
import pandas as pd

from typing import Dict, List, Tuple
from scipy.stats import spearmanr, pearsonr, t
from sklearn.linear_model import LinearRegression

# Try to import modern SciPy binomtest; fallback to statsmodels or manual exact if unavailable
try:
    from scipy.stats import binomtest as _binomtest
    def exact_binom_p(k, n, p=0.5, alternative='two-sided'):
        res = _binomtest(k, n, p, alternative=alternative)
        return float(res.pvalue)
except Exception:
    try:
        from statsmodels.stats.proportion import binom_test as _sm_binom_test
        def exact_binom_p(k, n, p=0.5, alternative='two-sided'):
            return float(_sm_binom_test(k, n, p, alternative=alternative))
    except Exception:
        # Manual exact two-sided p-value using pmf summation
        from math import comb
        def _pmf(k, n, p):
            return comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
        def exact_binom_p(k, n, p=0.5, alternative='two-sided'):
            # Two-sided p: sum of probabilities of outcomes with pmf <= pmf(k)
            p_k = _pmf(k, n, p)
            return float(sum(_pmf(i, n, p) for i in range(n + 1) if _pmf(i, n, p) <= p_k))


# =========================
# Configuration
# =========================
DIMENSIONS = ['PDI', 'IDV', 'MAS', 'UAI', 'LTO', 'IVR']

# VSM 2013 formulas: dimension = Σ weight * (m[Qa] - m[Qb]) ; constants C are set to 0 for structural comparison
DIM_Q = {
    'PDI': [('Q07', 'Q02', 35), ('Q20', 'Q23', 25)],
    'IDV': [('Q04', 'Q01', 35), ('Q09', 'Q06', 35)],
    'MAS': [('Q05', 'Q03', 35), ('Q08', 'Q10', 35)],
    'UAI': [('Q18', 'Q15', 40), ('Q21', 'Q24', 25)],
    'LTO': [('Q13', 'Q14', 40), ('Q19', 'Q22', 25)],
    'IVR': [('Q12', 'Q11', 35), ('Q17', 'Q16', 40)]
}

# Thresholds and permutation settings
TAU_H_QUANTILE = 0.0       # e.g., 0.0 (no threshold), 0.25, or 0.4
C_MULTIPLIER = 1.0         # τ_L = C_MULTIPLIER * SE_ΔL
N_PERMUTATIONS = 10000
RANDOM_SEED = 20250909

np.random.seed(RANDOM_SEED)

# Multiple-comparison correction settings
# method: 'holm' (default) | 'bh' (Benjamini-Hochberg FDR) | None
MTC_METHOD = 'holm'
MTC_ALPHA = 0.05
# Which tiers should apply across-model correction: Tier1 (binomial), Tier2 (permutation Spearman), Tier4 (permutation Pearson)
APPLY_MTC_TIER1 = True
APPLY_MTC_TIER2 = True
APPLY_MTC_TIER4 = True
# Whether to additionally write an adjusted-results copy
WRITE_CORRECTED_COPY = True
CORRECTED_FILE_NAME = 'tier_results_corrected.json'


# =========================
# Utilities
# =========================
REGION_MAINLAND = "Mainland"
REGION_HONG_KONG = "Hong Kong"

_REGION_ALIASES = {
    "\u5185\u5730": REGION_MAINLAND,
    "\u9999\u6e2f": REGION_HONG_KONG,
    "cn": REGION_MAINLAND,
    "hk": REGION_HONG_KONG,
    "mainland": REGION_MAINLAND,
    "hongkong": REGION_HONG_KONG,
    "hong kong": REGION_HONG_KONG,
}


def normalize_region_value(value):
    if pd.isna(value):
        return value
    s = str(value).strip()
    key = s.lower().replace("_", "").replace("-", "").replace(" ", "")
    if s in _REGION_ALIASES:
        return _REGION_ALIASES[s]
    if key in _REGION_ALIASES:
        return _REGION_ALIASES[key]
    return s


def normalize_region_series(series: pd.Series) -> pd.Series:
    return series.apply(normalize_region_value)


def load_items_long(csv_path='llm_vsm_items_long.csv') -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    expected = {'model', 'region', 'persona', 'item_id', 'response'}
    if not expected.issubset(df.columns):
        raise ValueError(f"Input file is missing required columns: {expected}")
    df['region'] = normalize_region_series(df['region'])
    df['response'] = pd.to_numeric(df['response'], errors='coerce')
    df = df.dropna(subset=['response'])
    return df


def load_official_deltaH(meta_path='preprocess_meta.json') -> Dict[str, float]:
    # Try read from preprocess_meta.json
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        if 'delta_h_official' in meta:
            return {k: float(v) for k, v in meta['delta_h_official'].items()}
    except Exception:
        pass
    # Fallback hard-coded
    HOFSTEDE_SCORES = {
        REGION_MAINLAND: {'PDI': 80, 'IDV': 43, 'MAS': 66, 'UAI': 30, 'LTO': 77, 'IVR': 24},
        REGION_HONG_KONG: {'PDI': 68, 'IDV': 50, 'MAS': 57, 'UAI': 29, 'LTO': 93, 'IVR': 17}
    }
    return {d: HOFSTEDE_SCORES[REGION_MAINLAND][d] - HOFSTEDE_SCORES[REGION_HONG_KONG][d] for d in DIMENSIONS}


def compute_item_stats(items_long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns per model-region-item stats: mean m, std sd, count n, se.
    """
    g = items_long_df.groupby(['model', 'region', 'item_id'])['response']
    stats = g.agg(['mean', 'std', 'count']).reset_index().rename(columns={'mean': 'm', 'std': 'sd', 'count': 'n'})
    stats['se'] = stats['sd'] / np.sqrt(stats['n'].replace(0, np.nan))
    return stats


def compute_vsm_indices(item_means_df: pd.DataFrame) -> pd.DataFrame:
    """
    item_means_df: ['model','region','item_id','m']
    Returns: ['model','region'] + indices.
    """
    pivot = item_means_df.pivot_table(index=['model', 'region'], columns='item_id', values='m')
    pivot = pivot.reset_index()

    rows = []
    for _, row in pivot.iterrows():
        rec = {'model': row['model'], 'region': row['region']}
        for dim in DIMENSIONS:
            score = 0.0
            for qa, qb, w in DIM_Q[dim]:
                ma = row.get(qa, np.nan)
                mb = row.get(qb, np.nan)
                score += w * (ma - mb)
            rec[dim] = score
        rows.append(rec)
    return pd.DataFrame(rows)


def compute_vsm_indices_se(item_stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    item_stats_df: ['model','region','item_id','m','sd','n','se']
    Returns: ['model','region','SE_PDI',...]
    """
    pivot_se = item_stats_df.pivot_table(index=['model', 'region'], columns='item_id', values='se')
    pivot_se = pivot_se.reset_index()

    rows = []
    for _, row in pivot_se.iterrows():
        rec = {'model': row['model'], 'region': row['region']}
        for dim in DIMENSIONS:
            var_dim = 0.0
            var_nan = False
            for qa, qb, w in DIM_Q[dim]:
                sea = row.get(qa, np.nan)
                seb = row.get(qb, np.nan)
                term = 0.0
                if pd.notna(sea):
                    term += (w ** 2) * (sea ** 2)
                else:
                    var_nan = True
                if pd.notna(seb):
                    term += (w ** 2) * (seb ** 2)
                else:
                    var_nan = True
                var_dim += term
            rec[f'SE_{dim}'] = np.sqrt(var_dim) if not var_nan else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def compute_deltaL(indices_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    indices_df: ['model','region', DIMENSIONS...]
    Returns: dict[model] -> {dim: ΔL}
    """
    out = {}
    for model in indices_df['model'].unique():
        sub = indices_df[indices_df['model'] == model]
        sub = sub.copy()
        sub['region'] = normalize_region_series(sub['region'])
        if not {REGION_MAINLAND, REGION_HONG_KONG}.issubset(set(sub['region'])):
            continue
        cn = sub[sub['region'] == REGION_MAINLAND].iloc[0]
        hk = sub[sub['region'] == REGION_HONG_KONG].iloc[0]
        out[model] = {dim: float(cn[dim] - hk[dim]) for dim in DIMENSIONS}
    return out


def compute_deltaL_se(indices_se_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    indices_se_df: ['model','region','SE_*']
    Returns: dict[model] -> {dim: SE_ΔL}
    """
    out = {}
    for model in indices_se_df['model'].unique():
        sub = indices_se_df[indices_se_df['model'] == model]
        sub = sub.copy()
        sub['region'] = normalize_region_series(sub['region'])
        if not {REGION_MAINLAND, REGION_HONG_KONG}.issubset(set(sub['region'])):
            continue
        cn = sub[sub['region'] == REGION_MAINLAND].iloc[0]
        hk = sub[sub['region'] == REGION_HONG_KONG].iloc[0]
        out[model] = {}
        for dim in DIMENSIONS:
            se_cn = cn.get(f'SE_{dim}', np.nan)
            se_hk = hk.get(f'SE_{dim}', np.nan)
            if pd.notna(se_cn) and pd.notna(se_hk):
                out[model][dim] = float(np.sqrt(se_cn ** 2 + se_hk ** 2))
            else:
                out[model][dim] = float('nan')
    return out


# =========================
# Tier tests
# =========================
def tier1_directional_binom(deltaH: Dict[str, float],
                            deltaL: Dict[str, float],
                            se_deltaL: Dict[str, float],
                            tau_H_quantile: float = 0.0,
                            c_multiplier: float = 1.0,
                            use_two_sided: bool = True) -> Dict:
    """
    Tier 1: Directional agreement under dual thresholds.
    - Eligible dimensions set S = { d | |ΔH_d| >= τ_H and |ΔL_d| >= τ_L_d }, where τ_L_d = c*SE_ΔL_d
    - Test proportion of sign matches against 0.5 using exact binomial test.

    Returns:
      dict with:
        eligible_dims, k (matches), n (eligible), acc, p_value, tau_H, c, per_dim table
    """
    # Compute tau_H based on quantile of |ΔH|
    abs_dh = np.array([abs(deltaH[d]) for d in DIMENSIONS], dtype=float)
    tau_H = float(np.nanquantile(abs_dh, tau_H_quantile)) if len(abs_dh) else 0.0

    per_dim = []
    k = 0
    n = 0
    for d in DIMENSIONS:
        dh = deltaH.get(d, np.nan)
        dl = deltaL.get(d, np.nan)
        se = se_deltaL.get(d, np.nan)
        if np.isnan(dh) or np.isnan(dl) or np.isnan(se):
            per_dim.append((d, dh, dl, se, False, None))
            continue
        tau_L = c_multiplier * se
        eligible = (abs(dh) >= tau_H) and (abs(dl) >= tau_L)
        if eligible:
            n += 1
            match = int(np.sign(dh) == np.sign(dl) and np.sign(dh) != 0 and np.sign(dl) != 0)
            k += match
            per_dim.append((d, dh, dl, se, True, match))
        else:
            per_dim.append((d, dh, dl, se, False, None))

    p_value = None
    if n > 0:
        alt = 'two-sided' if use_two_sided else 'greater'
        p_value = exact_binom_p(k, n, 0.5, alternative=alt)
    acc = (k / n) if n > 0 else np.nan

    return {
        "tau_H": tau_H,
        "c": c_multiplier,
        "eligible_dims": [d for (d, dh, dl, se, elg, m) in per_dim if elg],
        "k_matches": int(k),
        "n_eligible": int(n),
        "accuracy": float(acc) if not np.isnan(acc) else None,
        "p_value_binom": p_value,
        "per_dimension": [
            {
                "dimension": d,
                "deltaH": float(dh) if not np.isnan(dh) else None,
                "deltaL": float(dl) if not np.isnan(dl) else None,
                "SE_deltaL": float(se) if not np.isnan(se) else None,
                "eligible": bool(elg),
                "sign_match": (bool(m) if m is not None else None)
            } for (d, dh, dl, se, elg, m) in per_dim
        ]
    }


def permutation_p_for_correlation(x: np.ndarray, y: np.ndarray, kind: str = 'spearman',
                                  n_perm: int = 10000) -> Tuple[float, float]:
    """
    Compute permutation p-value for correlation by shuffling y.
    Returns (observed_r, p_value).
    """
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return np.nan, np.nan

    if kind == 'spearman':
        r_obs = spearmanr(x, y)[0]
    else:
        r_obs = pearsonr(x, y)[0]

    count = 0
    for _ in range(n_perm):
        y_perm = np.random.permutation(y)
        r_perm = spearmanr(x, y_perm)[0] if kind == 'spearman' else pearsonr(x, y_perm)[0]
        if abs(r_perm) >= abs(r_obs):
            count += 1
    p_val = (count + 1) / (n_perm + 1)  # add-one smoothing
    return float(r_obs), float(p_val)


def tier2_spearman(deltaH: Dict[str, float], deltaL: Dict[str, float], n_perm: int = 10000) -> Dict:
    x = np.array([deltaL[d] for d in DIMENSIONS], dtype=float)
    y = np.array([deltaH[d] for d in DIMENSIONS], dtype=float)
    r_s, p_perm = permutation_p_for_correlation(x, y, kind='spearman', n_perm=n_perm)
    # Also report asymptotic p (for reference)
    mask = ~np.isnan(x) & ~np.isnan(y)
    p_asym = float(spearmanr(x[mask], y[mask])[1]) if mask.sum() >= 3 else None
    return {"spearman_r": r_s, "p_perm": p_perm, "p_asym": p_asym}


def tier4_pearson(deltaH: Dict[str, float], deltaL: Dict[str, float], n_perm: int = 10000) -> Dict:
    x = np.array([deltaL[d] for d in DIMENSIONS], dtype=float)
    y = np.array([deltaH[d] for d in DIMENSIONS], dtype=float)
    r_p, p_perm = permutation_p_for_correlation(x, y, kind='pearson', n_perm=n_perm)
    # Asymptotic p for Pearson
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() >= 3:
        r, p_asym = pearsonr(x[mask], y[mask])
        p_asym = float(p_asym)
    else:
        p_asym = None
    return {"pearson_r": r_p, "p_perm": p_perm, "p_asym": p_asym}


def tier3_regression(deltaH: Dict[str, float], deltaL: Dict[str, float]) -> Dict:
    """
    OLS regression ΔH ≈ a·ΔL (+ b) with 6 points (dimensions).
    Reports:
      - No-intercept model: slope a0, R²_0, t-test & 95% CI for a0
      - With-intercept model: slope a1, intercept b1, R²_1, t-test & 95% CI for a1
      - Leave-One-Dimension-Out (LODO-CV) average R² for both models
    """
    dims = DIMENSIONS
    x = np.array([deltaL[d] for d in dims], dtype=float).reshape(-1, 1)
    y = np.array([deltaH[d] for d in dims], dtype=float)

    # Helper to compute CI for slope
    def slope_ci_no_intercept(x, y, alpha=0.05):
        n = len(y)
        x_flat = x.reshape(-1)
        a = np.sum(x_flat * y) / np.sum(x_flat ** 2)
        y_hat = a * x_flat
        resid = y - y_hat
        dof = n - 1
        if dof <= 0:
            return a, np.nan, np.nan, np.nan, np.nan, np.nan
        s2 = np.sum(resid ** 2) / dof
        se_a = math.sqrt(s2 / np.sum(x_flat ** 2))
        t_stat = a / se_a if se_a > 0 else np.nan
        t_crit = t.ppf(1 - 0.05 / 2, dof)
        ci_low = a - t_crit * se_a
        ci_high = a + t_crit * se_a
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum(resid ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return a, se_a, t_stat, ci_low, ci_high, r2

    def slope_ci_with_intercept(x, y, alpha=0.05):
        n = len(y)
        X = np.hstack([np.ones((n, 1)), x])
        model = LinearRegression().fit(X, y)
        b = model.intercept_
        a = model.coef_[1]
        y_hat = model.predict(X)
        resid = y - y_hat
        dof = n - 2
        if dof <= 0:
            return a, b, np.nan, np.nan, np.nan, np.nan, np.nan
        s2 = np.sum(resid ** 2) / dof
        XtX_inv = np.linalg.inv(X.T @ X)
        se_a = math.sqrt(s2 * XtX_inv[1, 1])
        t_stat = a / se_a if se_a > 0 else np.nan
        t_crit = t.ppf(1 - 0.05 / 2, dof)
        ci_low = a - t_crit * se_a
        ci_high = a + t_crit * se_a
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum(resid ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return a, b, se_a, t_stat, ci_low, ci_high, r2

    a0, se0, t0, ci0_low, ci0_high, r2_0 = slope_ci_no_intercept(x, y)
    a1, b1, se1, t1, ci1_low, ci1_high, r2_1 = slope_ci_with_intercept(x, y)

    # LODO-CV R² (mean training R² as a robustness proxy)
    def lodo_r2(no_intercept: bool) -> float:
        r2_list = []
        for i in range(len(dims)):
            mask = np.ones(len(dims), dtype=bool)
            mask[i] = False
            x_tr, y_tr = x[mask], y[mask]
            if no_intercept:
                x_flat = x_tr.reshape(-1)
                a = np.sum(x_flat * y_tr) / np.sum(x_flat ** 2)
                y_hat = a * x_flat
                ss_tot = np.sum((y_tr - np.mean(y_tr)) ** 2)
                ss_res = np.sum((y_tr - y_hat) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            else:
                Xtr = np.hstack([np.ones((len(y_tr), 1)), x_tr])
                mdl = LinearRegression().fit(Xtr, y_tr)
                y_hat = mdl.predict(Xtr)
                ss_tot = np.sum((y_tr - np.mean(y_tr)) ** 2)
                ss_res = np.sum((y_tr - y_hat) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            r2_list.append(r2)
        return float(np.nanmean(r2_list))

    r2_0_lodo = lodo_r2(no_intercept=True)
    r2_1_lodo = lodo_r2(no_intercept=False)

    return {
        "no_intercept": {
            "slope_a": float(a0) if not np.isnan(a0) else None,
            "SE_a": float(se0) if not np.isnan(se0) else None,
            "t_stat": float(t0) if not np.isnan(t0) else None,
            "CI_95": [float(ci0_low) if not np.isnan(ci0_low) else None,
                      float(ci0_high) if not np.isnan(ci0_high) else None],
            "R2": float(r2_0) if not np.isnan(r2_0) else None,
            "R2_LODO": float(r2_0_lodo) if not np.isnan(r2_0_lodo) else None
        },
        "with_intercept": {
            "slope_a": float(a1) if not np.isnan(a1) else None,
            "intercept_b": float(b1) if not np.isnan(b1) else None,
            "SE_a": float(se1) if not np.isnan(se1) else None,
            "t_stat": float(t1) if not np.isnan(t1) else None,
            "CI_95": [float(ci1_low) if not np.isnan(ci1_low) else None,
                      float(ci1_high) if not np.isnan(ci1_high) else None],
            "R2": float(r2_1) if not np.isnan(r2_1) else None,
            "R2_LODO": float(r2_1_lodo) if not np.isnan(r2_1_lodo) else None
        }
    }


# =========================
# Multiple-comparison correction
# =========================
def _holm_adjust(pvals: List[float]) -> List[float]:
    """Holm-Bonferroni: returns adjusted p-values in original order (monotone)."""
    m = len(pvals)
    if m <= 1:
        return pvals[:]
    order = np.argsort(pvals)
    p_sorted = np.array(pvals)[order]
    # Holm: adj_i = max_{j<=i} ( (m-j) * p_sorted[j] ) (1-based description, implemented in 0-based)
    adj_sorted = np.empty(m, dtype=float)
    running_max = 0.0
    for i in range(m):
        factor = m - i
        val = factor * p_sorted[i]
        running_max = max(running_max, val)
        adj_sorted[i] = min(running_max, 1.0)
    # Restore original order
    adj = np.empty(m, dtype=float)
    adj[order] = adj_sorted
    return adj.tolist()


def _bh_adjust(pvals: List[float]) -> List[float]:
    """Benjamini-Hochberg FDR: returns q-values in original order (monotone)."""
    m = len(pvals)
    if m <= 1:
        return pvals[:]
    order = np.argsort(pvals)
    p_sorted = np.array(pvals)[order]
    q_sorted = np.empty(m, dtype=float)
    running_min = 1.0
    for i in range(m-1, -1, -1):
        factor = (m / (i+1)) * p_sorted[i]
        running_min = min(running_min, factor)
        q_sorted[i] = min(running_min, 1.0)
    q = np.empty(m, dtype=float)
    q[order] = q_sorted
    return q.tolist()


def adjust_family_pvalues(p_map: Dict[str, float], method: str = 'holm') -> Dict[str, float]:
    """
    p_map: {model: p_raw or None}
    return: {model: p_adj} (adjusts non-None entries only)
    """
    # Filter invalid
    keys = [k for k, v in p_map.items() if v is not None and not np.isnan(v)]
    vals = [p_map[k] for k in keys]
    if len(vals) <= 1 or method is None:
        return {k: p_map.get(k, None) for k in p_map}
    if method.lower() == 'holm':
        adj_vals = _holm_adjust(vals)
    elif method.lower() in ('bh', 'fdr', 'bh-fdr'):
        adj_vals = _bh_adjust(vals)
    else:
        # Unrecognized method: return original values
        return {k: p_map.get(k, None) for k in p_map}
    return {k: float(a) for k, a in zip(keys, adj_vals)}


# =========================
# Main orchestration
# =========================
def main():
    print("--- VSM 2013 indices + ΔL computation + four-tier validation (with multiple-comparison correction) ---")

    # Load data
    items_long = load_items_long('llm_vsm_items_long.csv')
    deltaH_official = load_official_deltaH('preprocess_meta.json')

    # Compute item stats and indices
    item_stats = compute_item_stats(items_long)
    item_stats[['model','region','item_id','m']].to_csv('vsm_item_means.csv', index=False)
    print("✅ Wrote: vsm_item_means.csv")

    indices_df = compute_vsm_indices(item_stats[['model','region','item_id','m']])
    indices_df.to_csv('vsm_indices_by_model_region.csv', index=False)
    print("✅ Wrote: vsm_indices_by_model_region.csv")

    indices_se_df = compute_vsm_indices_se(item_stats)

    # Delta L and its SE
    deltaL_by_model = compute_deltaL(indices_df)
    deltaL_se_by_model = compute_deltaL_se(indices_se_df)

    with open('deltaL_by_model.json', 'w', encoding='utf-8') as f:
        json.dump(deltaL_by_model, f, ensure_ascii=False, indent=2)
    with open('deltaH_official.json', 'w', encoding='utf-8') as f:
        json.dump(deltaH_official, f, ensure_ascii=False, indent=2)
    print("✅ Wrote: deltaL_by_model.json, deltaH_official.json")

    # Full Tier results per model
    all_results = {}
    for model, dL in deltaL_by_model.items():
        seL = deltaL_se_by_model.get(model, {})
        # Tier 1
        t1 = tier1_directional_binom(
            deltaH=deltaH_official,
            deltaL=dL,
            se_deltaL=seL,
            tau_H_quantile=TAU_H_QUANTILE,
            c_multiplier=C_MULTIPLIER,
            use_two_sided=True
        )
        # Tier 2
        t2 = tier2_spearman(deltaH_official, dL, n_perm=N_PERMUTATIONS)
        # Tier 3
        t3 = tier3_regression(deltaH_official, dL)
        # Tier 4
        t4 = tier4_pearson(deltaH_official, dL, n_perm=N_PERMUTATIONS)

        all_results[model] = {
            "Tier1_directional_binomial": t1,
            "Tier2_spearman": t2,
            "Tier3_regression": t3,
            "Tier4_pearson": t4,
            "DeltaL": dL,
            "SE_DeltaL": seL,
            "DeltaH_official": deltaH_official,
            "config": {
                "tau_H_quantile": TAU_H_QUANTILE,
                "c_multiplier": C_MULTIPLIER,
                "n_permutations": N_PERMUTATIONS,
                "random_seed": RANDOM_SEED
            }
        }

    # ============ Multiple-comparison correction across models ============
    # Family 1: Tier1 (binom)
    # Family 2: Tier2 (spearman p_perm)
    # Family 3: Tier4 (pearson p_perm)
    mtc_summary = {
        "method": MTC_METHOD,
        "alpha": MTC_ALPHA,
        "families": {}
    }

    models = list(all_results.keys())

    # Tier1 family
    if APPLY_MTC_TIER1 and MTC_METHOD:
        pmap_t1 = {m: all_results[m]["Tier1_directional_binomial"].get("p_value_binom") for m in models}
        padj_t1 = adjust_family_pvalues(pmap_t1, method=MTC_METHOD)
        for m in models:
            if pmap_t1[m] is not None:
                all_results[m]["Tier1_directional_binomial"]["p_value_binom_adj"] = float(padj_t1[m])
                all_results[m]["Tier1_directional_binomial"]["significant_adj"] = (padj_t1[m] < MTC_ALPHA)
        mtc_summary["families"]["Tier1_binomial"] = {
            m: {"raw": pmap_t1[m], "adj": padj_t1[m]} for m in models
        }

    # Tier2 family
    if APPLY_MTC_TIER2 and MTC_METHOD:
        pmap_t2 = {m: all_results[m]["Tier2_spearman"].get("p_perm") for m in models}
        padj_t2 = adjust_family_pvalues(pmap_t2, method=MTC_METHOD)
        for m in models:
            if pmap_t2[m] is not None:
                all_results[m]["Tier2_spearman"]["p_perm_adj"] = float(padj_t2[m])
                all_results[m]["Tier2_spearman"]["significant_adj"] = (padj_t2[m] < MTC_ALPHA)
        mtc_summary["families"]["Tier2_spearman_perm"] = {
            m: {"raw": pmap_t2[m], "adj": padj_t2[m]} for m in models
        }

    # Tier4 family
    if APPLY_MTC_TIER4 and MTC_METHOD:
        pmap_t4 = {m: all_results[m]["Tier4_pearson"].get("p_perm") for m in models}
        padj_t4 = adjust_family_pvalues(pmap_t4, method=MTC_METHOD)
        for m in models:
            if pmap_t4[m] is not None:
                all_results[m]["Tier4_pearson"]["p_perm_adj"] = float(padj_t4[m])
                all_results[m]["Tier4_pearson"]["significant_adj"] = (padj_t4[m] < MTC_ALPHA)
        mtc_summary["families"]["Tier4_pearson_perm"] = {
            m: {"raw": pmap_t4[m], "adj": padj_t4[m]} for m in models
        }

    # Attach correction summary at the top level
    all_results["_multiple_comparison_correction"] = mtc_summary

    with open('tier_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print("✅ Wrote: tier_results.json (includes multiple-comparison correction fields)")

    if WRITE_CORRECTED_COPY:
        with open(CORRECTED_FILE_NAME, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"✅ Wrote: {CORRECTED_FILE_NAME}")

    # Console summary
    print("\nBaseline ΔH (Mainland − Hong Kong):", deltaH_official)

    # Iterate only over actual model keys (avoid treating _multiple_comparison_correction as a model)
    for model in models:
        res = all_results[model]
        print(f"\n=== Model {model} summary ===")

        t1 = res["Tier1_directional_binomial"]
        acc = t1['accuracy']
        acc_str = f"{acc:.2f}" if acc is not None else "NA"
        p1 = t1.get('p_value_binom')
        p1_adj = t1.get('p_value_binom_adj')
        print(f"Tier1: directional agreement={acc_str} (k={t1['k_matches']}, n={t1['n_eligible']}) "
              f"p_exact={p1}, p_adj={p1_adj}, tau_H={t1['tau_H']:.3f}, c={t1['c']}")

        t2 = res["Tier2_spearman"]
        print(f"Tier2: Spearman r={t2['spearman_r']:.3f}, p_perm={t2['p_perm']}, "
              f"p_adj={t2.get('p_perm_adj')}, p_asym={t2['p_asym']}")

        t3 = res["Tier3_regression"]
        print(f"Tier3: no-intercept a={t3['no_intercept']['slope_a']:.3f}, R2={t3['no_intercept']['R2']:.3f}, "
              f"CI95={t3['no_intercept']['CI_95']}")
        print(f"       with-intercept a={t3['with_intercept']['slope_a']:.3f}, b={t3['with_intercept']['intercept_b']:.3f}, "
              f"R2={t3['with_intercept']['R2']:.3f}, CI95={t3['with_intercept']['CI_95']}")

        t4 = res["Tier4_pearson"]
        print(f"Tier4: Pearson r={t4['pearson_r']:.3f}, p_perm={t4['p_perm']}, "
              f"p_adj={t4.get('p_perm_adj')}, p_asym={t4['p_asym']}")

              
    print("\nNotes:")
    print("- Tier1 uses an exact binomial test; eligible dims require |ΔH|≥τ_H and |ΔL|≥c·SE_ΔL.")
    print("- Tier2/Tier4 use permutation tests (default 10000) and apply cross-model multiple-comparison correction.")
    print("- Tier3 reports slopes/CI/R² for no-intercept and intercept models, plus a LODO robustness proxy.")
    print(f"- Multiple-comparison method: {MTC_METHOD}; alpha={MTC_ALPHA}. Results are written back and summarized in _multiple_comparison_correction.")
    print("- Direction is fixed as Mainland − Hong Kong.")


if __name__ == "__main__":
    main()
