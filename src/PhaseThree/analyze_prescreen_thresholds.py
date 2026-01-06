# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt

INFILES = ['screening_log_clean.csv', 'screening_log.csv']
OUT_SUMMARY = 'prescreen_summary_thresholds.csv'
OUT_TOP1 = 'top1_by_threshold.txt'
OUTDIR_FIG = 'prescreen_figs'
os.makedirs(OUTDIR_FIG, exist_ok=True)

# Set grouping granularity
# To include region, set INCLUDE_REGION = True
INCLUDE_REGION = False

# Score column priority
SCORE_COLS = ['initial_weighted_total', 'AI_total', 'Human1_total', 'Human2_total']

# Threshold band (±)
BANDS = [0.5, 1.0]

def to_float(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().replace('?', '').replace('，','.')
    try:
        return float(s)
    except:
        return np.nan

def norm_bool(x):
    if pd.isna(x): return None
    s = str(x).strip().upper()
    if s in ['TRUE','T','YES','Y','1']: return True
    if s in ['FALSE','F','NO','N','0']: return False
    if s == 'TURE': return True
    return None

def map_method(m):
    if pd.isna(m): return m
    s = str(m)
    s = s.replace('\u89c4\u5219\u89e3\u7801\u9a71\u52a8\u4eba\u8bbe\u5408\u6210', 'RDPS')
    s = s.replace('\u6c89\u6d78\u5f0f\u89d2\u8272\u626e\u6f14\u4eba\u8bbe\u751f\u6210', 'IRPG')
    s = s.replace('\u81ea\u63a8\u65ad\u4eba\u8bbe\u751f\u6210', 'AIPG')
    return s

def parse_variant(prompt):
    p = str(prompt).lower()
    tags = []
    if 'normal' in p: tags.append('Normal')
    if "era's features language distinction" in p or 'era' in p:
        tags.append('Era')
    if 'cantonese' in p: tags.append('Cantonese')
    return '+'.join(tags) if tags else 'Normal'

def load_df():
    df = None
    for f in INFILES:
        if os.path.exists(f):
            try:
                df = pd.read_csv(f, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(f, encoding='gbk')
            break
    if df is None:
        raise FileNotFoundError('Could not find screening_log_clean.csv or screening_log.csv')

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    # Fix key columns
    if 'method' in df.columns:
        df['method'] = df['method'].map(map_method)
    else:
        df['method'] = 'UNKNOWN'

    if 'prompt' in df.columns:
        df['variant'] = df['prompt'].map(parse_variant)
    else:
        df['variant'] = 'Normal'

    if 'language' in df.columns:
        df['language'] = df['language'].map(lambda s: str(s).capitalize() if pd.notna(s) else s)
    else:
        df['language'] = 'Unknown'

    if 'region' not in df.columns:
        df['region'] = 'Unknown'

    # Clean score columns
    for col in SCORE_COLS:
        if col in df.columns:
            df[col] = df[col].map(to_float)

    # Construct composite score
    def pick_score(row):
        for c in SCORE_COLS:
            if c in row and pd.notna(row[c]):
                return row[c]
        return np.nan
    df['score'] = df.apply(pick_score, axis=1)

    # Range check
    MAXSCORE = 25.0
    mask_bad = (df['score'].notna()) & ((df['score'] < 0) | (df['score'] > MAXSCORE))
    if mask_bad.any():
        print(f'Warning: score has {int(mask_bad.sum())} out-of-range values; setting to NaN.')
        df.loc[mask_bad, 'score'] = np.nan

    # Drop rows without score
    df = df[df['score'].notna()].copy()
    return df

def group_keys(df):
    keys = ['method', 'variant', 'language']
    if INCLUDE_REGION:
        keys.insert(2, 'region')  # method, variant, region, language
    return keys

def compute_pass_rates(df):
    keys = group_keys(df)
    # Use group mean as the center threshold
    grp = df.groupby(keys, dropna=False)
    centers = grp['score'].mean().rename('center').reset_index()
    df2 = df.merge(centers, on=keys, how='left')
    # Compute deviation
    df2['dev'] = (df2['score'] - df2['center']).abs()

    rows = []
    for band in BANDS:
        # Pass if within band
        df2[f'pass_{band}'] = df2['dev'] <= band
        g = df2.groupby(keys, dropna=False)
        agg = g.agg(
            n=('score','count'),
            pass_n=(f'pass_{band}', 'sum'),
            mean_score=('score','mean'),
            std_score=('score','std'),
            center=('center','mean')
        ).reset_index()
        agg['band'] = band
        agg['pass_rate'] = agg['pass_n'] / agg['n']
        rows.append(agg)
    out = pd.concat(rows, ignore_index=True)
    return out

def pick_top1(summary):
    keys = group_keys(summary)
    res_lines = []
    for band in BANDS:
        sub = summary[summary['band']==band].copy()
        if sub.empty:
            res_lines.append(f'±{band}: no data')
            continue
        # Sort: pass_rate desc, mean_score desc, n desc
        sub = sub.sort_values(['pass_rate','mean_score','n'], ascending=[False,False,False])
        top1 = sub.iloc[0]
        ident = ' | '.join(f'{k}={top1[k]}' for k in keys)
        res_lines.append(f'±{band}: Top-1 -> {ident} | pass_rate={top1["pass_rate"]:.3f}, n={int(top1["n"])}, mean={top1["mean_score"]:.3f}')
    # Compare whether Top-1 changes across bands
    changed = 'unknown'
    if len(BANDS) >= 2:
        b0, b1 = BANDS[0], BANDS[1]
        sub0 = summary[summary['band']==b0].sort_values(['pass_rate','mean_score','n'], ascending=[False,False,False])
        sub1 = summary[summary['band']==b1].sort_values(['pass_rate','mean_score','n'], ascending=[False,False,False])
        if sub0.empty or sub1.empty:
            changed = 'unknown'
        else:
            keys = group_keys(summary)
            t0 = tuple(sub0.iloc[0][keys])
            t1 = tuple(sub1.iloc[0][keys])
            changed = 'changed' if t0 != t1 else 'unchanged'
        res_lines.append(f'Top-1 combo at ±{b0} vs ±{b1}: {changed}')
    return '\n'.join(res_lines), changed

def plot_pass_rates(summary):
    # One bar chart per band
    keys = group_keys(summary)
    label_col = 'combo'
    summary[label_col] = summary[keys].astype(str).agg(' | '.join, axis=1)
    for band in BANDS:
        sub = summary[summary['band']==band].copy()
        if sub.empty: continue
        sub = sub.sort_values('pass_rate', ascending=False)
        plt.figure(figsize=(10, max(3, 0.35*len(sub))))
        plt.barh(sub['combo'], sub['pass_rate'], color='#4C78A8')
        for i, v in enumerate(sub['pass_rate']):
            plt.text(v+0.005, i, f'{v:.2f}', va='center')
        plt.title(f'Pass rate within ±{band} of group mean')
        plt.xlabel('Pass rate')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        fname = os.path.join(OUTDIR_FIG, f'pass_rate_band_{band}.png')
        plt.savefig(fname, dpi=200)
        plt.close()
        print(f'Wrote figure: {fname}')

def main():
    df = load_df()
    summary = compute_pass_rates(df)
    summary.to_csv(OUT_SUMMARY, index=False, encoding='utf-8-sig')
    print('Wrote summary:', OUT_SUMMARY)

    top1_txt, changed = pick_top1(summary)
    with open(OUT_TOP1, 'w', encoding='utf-8') as f:
        f.write(top1_txt + '\n')
    print('Top-1 results:\n' + top1_txt)

    plot_pass_rates(summary)

if __name__ == '__main__':
    main()
