# -*- coding: utf-8 -*-
"""
Run VSM indices and four-tier tests grouped by condition. Outputs:
- tier_results_by_condition.json
- judgeable_set_summary_by_condition.csv
- vsm_indices_by_condition_model_region.csv
"""
import json
import numpy as np
import pandas as pd
import analysis_vsm_indices as base

DIMENSIONS = base.DIMENSIONS

def run_one_condition(df_cond: pd.DataFrame, deltaH_official: dict):
    df_cond = df_cond.copy()
    df_cond['region'] = base.normalize_region_series(df_cond['region'])
    regs = set(df_cond['region'].unique())
    if not {base.REGION_MAINLAND, base.REGION_HONG_KONG}.issubset(regs):
        return {}, pd.DataFrame()

    g = df_cond.groupby(['model','region','item_id'])['response']
    item_stats = g.agg(['mean','std','count']).reset_index().rename(columns={'mean':'m','std':'sd','count':'n'})
    item_stats['se'] = item_stats['sd'] / np.sqrt(item_stats['n'].replace(0, np.nan))

    indices_df = base.compute_vsm_indices(item_stats[['model','region','item_id','m']])
    indices_se_df = base.compute_vsm_indices_se(item_stats)

    deltaL_by_model = base.compute_deltaL(indices_df)
    deltaL_se_by_model = base.compute_deltaL_se(indices_se_df)

    results = {}
    for model, dL in deltaL_by_model.items():
        seL = deltaL_se_by_model.get(model, {})
        t1 = base.tier1_directional_binom(
            deltaH=deltaH_official, deltaL=dL, se_deltaL=seL,
            tau_H_quantile=base.TAU_H_QUANTILE, c_multiplier=base.C_MULTIPLIER, use_two_sided=True
        )
        t2 = base.tier2_spearman(deltaH_official, dL, n_perm=base.N_PERMUTATIONS)
        t3 = base.tier3_regression(deltaH_official, dL)
        t4 = base.tier4_pearson(deltaH_official, dL, n_perm=base.N_PERMUTATIONS)

        results[model] = {
            "Tier1_directional_binomial": t1,
            "Tier2_spearman": t2,
            "Tier3_regression": t3,
            "Tier4_pearson": t4,
            "DeltaL": dL,
            "SE_DeltaL": seL,
            "DeltaH_official": deltaH_official,
            "config": {
                "tau_H_quantile": base.TAU_H_QUANTILE,
                "c_multiplier": base.C_MULTIPLIER,
                "n_permutations": base.N_PERMUTATIONS,
                "random_seed": base.RANDOM_SEED
            }
        }
    return results, indices_df

def main():
    df = pd.read_csv('llm_vsm_items_long_condition.csv')
    need = {'condition','model','region','item_id','response'}
    if not need.issubset(df.columns):
        raise ValueError(f"Missing required columns: {need - set(df.columns)}")
    df = df[['condition','model','region','item_id','response']].copy()

    try:
        with open('preprocess_meta.json','r',encoding='utf-8') as f:
            meta = json.load(f)
            deltaH = meta.get('delta_h_official', None)
    except Exception:
        deltaH = None
    if deltaH is None:
        deltaH = base.load_official_deltaH('preprocess_meta.json')

    bycond = {}
    indices_rows = []
    for cond, sub in df.groupby('condition'):
        res, idx = run_one_condition(sub, deltaH)
        if not res:
            print(f"Skipping condition (missing one region in pair): {cond}")
            continue
        bycond[cond] = res
        idx2 = idx.copy(); idx2['condition'] = cond
        indices_rows.append(idx2)

    with open('tier_results_by_condition.json','w',encoding='utf-8') as f:
        json.dump(bycond, f, ensure_ascii=True, indent=2)
    print("Wrote tier_results_by_condition.json")

    if indices_rows:
        indices_all = pd.concat(indices_rows, ignore_index=True)
        indices_all.to_csv('vsm_indices_by_condition_model_region.csv', index=False, encoding='utf-8-sig')
        print("Wrote vsm_indices_by_condition_model_region.csv")
    else:
        print("No indices data available (all conditions may be missing paired regions).")

    rows = []
    for cond, models in bycond.items():
        for model, blk in models.items():
            t1 = blk['Tier1_directional_binomial']
            nJ = t1['n_eligible']
            per = t1['per_dimension']
            row = {'condition': cond, 'model': model, 'J_size': nJ}
            for d in DIMENSIONS:
                elg = next((x['eligible'] for x in per if x['dimension']==d), False)
                row[f'in_J_{d}'] = int(bool(elg))
            rows.append(row)
    dfJ = pd.DataFrame(rows)
    dfJ.to_csv('judgeable_set_summary_by_condition.csv', index=False, encoding='utf-8-sig')
    print("Wrote judgeable_set_summary_by_condition.csv")

if __name__ == '__main__':
    main()
