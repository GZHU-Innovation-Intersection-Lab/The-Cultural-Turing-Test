# -*- coding: utf-8 -*-
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

INFILE = 'prescreen_summary_thresholds.csv'
OUTDIR = 'figs'

def ensure_outdir(outdir: str = OUTDIR) -> str:
    os.makedirs(outdir, exist_ok=True)
    return outdir

def setup_plot_style():
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

# Parse long method strings into (Method, Locale, Era)
def parse_method(x):
    s = str(x)
    # Method
    if s.startswith('AIPG'):
        method = 'AIPG'
    elif s.startswith('IRPG'):
        method = 'IRPG'
    elif s.startswith('RDPS'):
        method = 'RDPS'
    else:
        method = s.split('-')[0]
    locale = 'Cantonese' if 'Cantonese' in s else ('Mandarin' if 'CN' in s or 'Mandarin' in s else '')
    era = 'Era' if 'Era' in s or "Era's Features" in s else 'Normal'
    return method, locale, era

def build_alias(row):
    m, loc, era = parse_method(row['method'])
    lang = row.get('language', '')
    if isinstance(lang, str) and lang:
        loc = lang
    return f'{m} | {loc} | {row["variant"]}'

def load_data():
    df = pd.read_csv(INFILE)
    for c in ['method','variant','language']:
        if c not in df.columns:
            raise ValueError(f'Missing column: {c}')
    df['alias'] = df.apply(build_alias, axis=1)
    df['Method'] = df['method'].apply(lambda x: parse_method(x)[0])
    return df

def plot_band(df, band_val, filename):
    setup_plot_style()
    ensure_outdir(OUTDIR)
    sub = df[df['band']==band_val].copy()
    if sub.empty: return
    sub = sub.sort_values(['pass_rate','mean_score'], ascending=[False,False])
    palette = {'AIPG':'#4C78A8','IRPG':'#F58518','RDPS':'#54A24B'}
    colors = sub['Method'].map(palette)
    height = max(3, 0.5*len(sub))
    plt.figure(figsize=(10, height))
    bars = plt.barh(sub['alias'], sub['pass_rate'], color=colors, edgecolor='none')
    for i, v in enumerate(sub['pass_rate']):
        plt.text(v + 0.01, i, f'{v:.2f}', va='center', fontsize=9)
    plt.gca().invert_yaxis()
    plt.xlabel(f'Pass rate (±{band_val} band)')
    plt.ylabel('Method | Locale | Variant')
    plt.title(f'Prescreen pass-rate comparison (±{band_val})')
    handles = [plt.Rectangle((0,0),1,1,color=palette[k]) for k in palette]
    labels = list(palette.keys())
    plt.legend(handles, labels, title='Method', loc='lower right', frameon=False)
    plt.xlim(0, 1.05)
    plt.grid(axis='x', alpha=0.2)
    plt.tight_layout()
    path = os.path.join(OUTDIR, filename)
    plt.savefig(path, dpi=220)
    plt.close()
    print('Wrote figure:', path)

def plot_by_method(df, band_val, method, filename):
    setup_plot_style()
    ensure_outdir(OUTDIR)
    sub = df[(df['band']==band_val) & (df['Method']==method)].copy()
    if sub.empty: return
    sub = sub.sort_values(['pass_rate','mean_score'], ascending=[False,False])
    height = max(2.5, 0.5*len(sub))
    plt.figure(figsize=(8, height))
    bars = plt.barh(sub['alias'], sub['pass_rate'], color='#4C78A8' if method=='AIPG' else ('#F58518' if method=='IRPG' else '#54A24B'))
    for i, v in enumerate(sub['pass_rate']):
        plt.text(v + 0.01, i, f'{v:.2f}', va='center', fontsize=9)
    plt.gca().invert_yaxis()
    plt.xlabel(f'Pass rate (±{band_val})')
    plt.ylabel('Method | Locale | Variant')
    plt.title(f'Pass rate for {method} at ±{band_val}')
    plt.xlim(0, 1.05)
    plt.grid(axis='x', alpha=0.2)
    plt.tight_layout()
    path = os.path.join(OUTDIR, filename)
    plt.savefig(path, dpi=220)
    plt.close()
    print('Wrote figure:', path)
