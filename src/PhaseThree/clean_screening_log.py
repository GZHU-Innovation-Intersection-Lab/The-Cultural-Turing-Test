# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re

INFILE = 'screening_log.csv'
OUTFILE = 'screening_log_clean.csv'

def to_float(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    s = s.replace('?', '')
    s = s.replace('，', '.')
    try:
        v = float(s)
        return v
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

def main():
    df = pd.read_csv(INFILE, encoding='gbk')
    df.columns = [c.strip() for c in df.columns]

    df['method'] = df['method'].map(map_method)

    df['variant'] = df['prompt'].map(parse_variant)

    for col in ['AI_total','Human1_total','Human2_total','initial_weighted_total']:
        if col in df.columns:
            df[col] = df[col].map(to_float)

    # Range check (assumes 0–25)
    MAXSCORE = 25.0
    for col in ['AI_total','Human1_total','Human2_total','initial_weighted_total']:
        if col in df.columns:
            mask_bad = (df[col].notna()) & ((df[col] < 0) | (df[col] > MAXSCORE))
            n_bad = int(mask_bad.sum())
            if n_bad:
                print(f'Warning: {col} has {n_bad} out-of-range values; set to NaN.')
                df.loc[mask_bad, col] = np.nan

    if 'passed_prescreen_base' in df.columns:
        df['passed_prescreen_base'] = df['passed_prescreen_base'].map(norm_bool)

    if 'language' in df.columns:
        df['language'] = df['language'].map(lambda s: str(s).capitalize() if pd.notna(s) else s)

    df.to_csv(OUTFILE, index=False, encoding='utf-8-sig')
    print('Wrote cleaned file:', OUTFILE)

if __name__ == '__main__':
    main()
