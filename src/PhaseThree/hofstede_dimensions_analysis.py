import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import LeaveOneOut
from sklearn.utils import resample
import json
import os
from typing import Dict, Tuple


REGION_MAINLAND = "Mainland"
REGION_HONGKONG = "Hong Kong"


HOFSTEDE_SCORES = {
    REGION_MAINLAND: {'PDI': 80, 'IDV': 43, 'MAS': 66, 'UAI': 30, 'LTO': 77, 'IVR': 24},
    REGION_HONGKONG: {'PDI': 68, 'IDV': 50, 'MAS': 57, 'UAI': 29, 'LTO': 93, 'IVR': 17}
}

DIMENSIONS = ['PDI', 'IDV', 'MAS', 'UAI', 'LTO', 'IVR']
MODELS = ['qwen3-max-preview']

N_BOOTSTRAPS = 10000
N_PERMUTATIONS = 10000
CONFIDENCE_LEVEL = 0.95
SE_C_VALUE = 1.64

DELTA_HOFSTEDE = {dim: HOFSTEDE_SCORES[REGION_MAINLAND][dim] - HOFSTEDE_SCORES[REGION_HONGKONG][dim] for dim in DIMENSIONS}


def load_excel_data(data_filepath: str) -> pd.DataFrame:
    all_data = []
    for model in MODELS:
        cn_file = os.path.join(data_filepath, f"CN-res({model}).xlsx")
        hk_file = os.path.join(data_filepath, f"HK-res({model}).xlsx")

        try:
            df_cn = pd.read_excel(cn_file, header=None)
            df_cn_melted = df_cn.melt(var_name='dimension_index', value_name='score')
            df_cn_melted['model'] = model
            df_cn_melted['region'] = REGION_MAINLAND
            df_cn_melted['dimension'] = df_cn_melted['dimension_index'].apply(
                lambda x: DIMENSIONS[x] if x < len(DIMENSIONS) else f'Unknown_{x}'
            )
            all_data.append(df_cn_melted[['model', 'region', 'dimension', 'score']])

            df_hk = pd.read_excel(hk_file, header=None)
            df_hk_melted = df_hk.melt(var_name='dimension_index', value_name='score')
            df_hk_melted['model'] = model
            df_hk_melted['region'] = REGION_HONGKONG
            df_hk_melted['dimension'] = df_hk_melted['dimension_index'].apply(
                lambda x: DIMENSIONS[x] if x < len(DIMENSIONS) else f'Unknown_{x}'
            )
            all_data.append(df_hk_melted[['model', 'region', 'dimension', 'score']])

        except FileNotFoundError as e:
            print(f"Warning: missing data file for model {model}: {e.filename}. Skipping.")
            continue
        except Exception as e:
            print(f"Error reading data for model {model}: {e}")
            continue

    if not all_data:
        raise ValueError("No model data loaded. Check the path and file format.")

    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df[combined_df['dimension'].isin(DIMENSIONS)]
    return combined_df


def calculate_delta_llm_with_se(data: pd.DataFrame) -> Dict[str, Tuple[Dict[str, float], Dict[str, float]]]:
    results = {}
    model_list = data['model'].unique()

    for model in model_list:
        model_df = data[data['model'] == model]

        agg_stats = model_df.groupby(['region', 'dimension'])['score'].agg(['mean', 'std', 'count']).reset_index()

        mainland_stats = agg_stats[agg_stats['region'] == REGION_MAINLAND].set_index('dimension')
        hongkong_stats = agg_stats[agg_stats['region'] == REGION_HONGKONG].set_index('dimension')

        if mainland_stats.empty or hongkong_stats.empty:
            print(f"Warning: incomplete data for model '{model}'. Missing {REGION_MAINLAND} or {REGION_HONGKONG}.")
            continue

        delta_l = (mainland_stats['mean'] - hongkong_stats['mean']).to_dict()

        se_mainland_sq = (mainland_stats['std'] ** 2) / mainland_stats['count']
        se_hongkong_sq = (hongkong_stats['std'] ** 2) / hongkong_stats['count']
        se_delta = np.sqrt(se_mainland_sq + se_hongkong_sq).to_dict()

        results[model] = (delta_l, se_delta)

    return results


def run_permutation_test(vec1: np.ndarray, vec2: np.ndarray, func) -> float:
    observed_stat = func(vec1, vec2)[0]
    perm_stats = []
    for _ in range(N_PERMUTATIONS):
        perm_vec2 = np.random.permutation(vec2)
        perm_stats.append(func(vec1, perm_vec2)[0])

    p_value = (np.sum(np.abs(perm_stats) >= np.abs(observed_stat)) + 1) / (N_PERMUTATIONS + 1)
    return p_value


def analysis_1_directional_agreement(delta_l: Dict, se_delta_l: Dict, delta_h: Dict, tau_h_quantile: float = 0.25) -> Dict:
    tau_h = np.percentile(np.abs(list(delta_h.values())), tau_h_quantile * 100)
    tau_l = {dim: SE_C_VALUE * se_delta_l.get(dim, 0) for dim in DIMENSIONS}

    judgeable_dims = [dim for dim in DIMENSIONS if abs(delta_h[dim]) >= tau_h and abs(delta_l[dim]) >= tau_l[dim]]

    if not judgeable_dims:
        return {"error": f"No judgeable dimensions (tau_h_quantile={tau_h_quantile})"}

    consistent_dims = [dim for dim in judgeable_dims if np.sign(delta_l[dim]) == np.sign(delta_h[dim])]

    n_consistent, n_judgeable = len(consistent_dims), len(judgeable_dims)
    accuracy = n_consistent / n_judgeable

    pval = stats.binom_test(n_consistent, n_judgeable, p=0.5, alternative='greater')

    return {
        f"accuracy_at_tau_h_q{int(tau_h_quantile * 100)}": accuracy,
        "n_consistent": n_consistent,
        "n_judgeable": n_judgeable,
        "p_value_binomial_test": pval,
        "is_significant": pval < (1 - CONFIDENCE_LEVEL),
        "judgeable_dimensions": judgeable_dims,
    }


def analysis_2_rank_order_agreement(delta_l: Dict, delta_h: Dict) -> Dict:
    vec_h, vec_l = np.array(list(delta_h.values())), np.array(list(delta_l.values()))

    if np.all(vec_h == vec_h[0]) or np.all(vec_l == vec_l[0]):
        return {
            "spearman_rho_signed": np.nan,
            "p_value_permutation": np.nan,
            "ci_signed": (np.nan, np.nan),
            "spearman_rho_absolute": np.nan,
            "p_value_permutation_abs": np.nan,
            "ci_absolute": (np.nan, np.nan),
            "warning": "One or both vectors are constant; correlation is undefined.",
        }

    rho_signed, _ = stats.spearmanr(vec_h, vec_l)
    rho_abs, _ = stats.spearmanr(np.abs(vec_h), np.abs(vec_l))

    p_perm_signed = run_permutation_test(vec_h, vec_l, stats.spearmanr)
    p_perm_abs = run_permutation_test(np.abs(vec_h), np.abs(vec_l), stats.spearmanr)

    indices = np.arange(len(DIMENSIONS))
    boot_rhos_signed = [stats.spearmanr(vec_h[idx], vec_l[idx])[0] for idx in [resample(indices) for _ in range(N_BOOTSTRAPS)]]
    boot_rhos_abs = [stats.spearmanr(np.abs(vec_h[idx]), np.abs(vec_l[idx]))[0] for idx in [resample(indices) for _ in range(N_BOOTSTRAPS)]]

    alpha = (1 - CONFIDENCE_LEVEL) / 2
    ci_signed = (np.percentile(boot_rhos_signed, alpha * 100), np.percentile(boot_rhos_signed, (1 - alpha) * 100))
    ci_abs = (np.percentile(boot_rhos_abs, alpha * 100), np.percentile(boot_rhos_abs, (1 - alpha) * 100))

    return {
        "spearman_rho_signed": rho_signed,
        "p_value_permutation": p_perm_signed,
        "ci_signed": ci_signed,
        "spearman_rho_absolute": rho_abs,
        "p_value_permutation_abs": p_perm_abs,
        "ci_absolute": ci_abs,
    }


def analysis_3_affine_alignment(delta_l: Dict, delta_h: Dict) -> Dict:
    X, y = np.array([list(delta_l.values())]).T, np.array(list(delta_h.values()))

    ols = LinearRegression().fit(X, y)

    boot_params = {'a': [], 'b': [], 'r2': []}
    indices = np.arange(len(DIMENSIONS))
    for _ in range(N_BOOTSTRAPS):
        boot_indices = resample(indices)
        boot_X, boot_y = X[boot_indices], y[boot_indices]
        if len(np.unique(boot_X)) < 2:
            continue
        boot_model = LinearRegression().fit(boot_X, boot_y)
        boot_params['a'].append(boot_model.coef_[0])
        boot_params['b'].append(boot_model.intercept_)
        boot_params['r2'].append(boot_model.score(boot_X, boot_y))

    alpha = (1 - CONFIDENCE_LEVEL) / 2
    ci = {k: (np.percentile(v, alpha * 100), np.percentile(v, (1 - alpha) * 100)) for k, v in boot_params.items()}

    loo, y_preds = LeaveOneOut(), []
    for train_idx, test_idx in loo.split(X):
        model_cv = LinearRegression().fit(X[train_idx], y[train_idx])
        y_preds.append(model_cv.predict(X[test_idx])[0])

    r2_lodo, mae_lodo = r2_score(y, y_preds), mean_absolute_error(y, y_preds)

    return {
        "ols_regression": {
            "r_squared": ols.score(X, y),
            "slope_a": ols.coef_[0],
            "intercept_b": ols.intercept_,
            "ci_r_squared": ci['r2'],
            "ci_slope_a": ci['a'],
            "ci_intercept_b": ci['b'],
            "is_slope_a_significant": ci['a'][0] > 0 or ci['a'][1] < 0,
        },
        "lodo_cross_validation": {"r_squared_cv": r2_lodo, "mae_cv": mae_lodo},
    }


def analysis_4_profile_similarity(delta_l: Dict, delta_h: Dict) -> Dict:
    vec_h, vec_l = np.array(list(delta_h.values())), np.array(list(delta_l.values()))

    pearson_r, _ = stats.pearsonr(vec_h, vec_l)
    p_perm_pearson = run_permutation_test(vec_h, vec_l, stats.pearsonr)

    cosine_sim = np.dot(vec_h, vec_l) / (np.linalg.norm(vec_h) * np.linalg.norm(vec_l))

    return {
        "description": "Supplementary holistic metrics.",
        "pearson_r": pearson_r,
        "p_value_permutation": p_perm_pearson,
        "cosine_similarity": cosine_sim,
    }


def main():
    print("--- Hofstede cultural difference analysis (robust, real-data loader) ---")

    data_directory = '.'

    try:
        raw_data = load_excel_data(data_directory)
        print(f"Loaded data. Total records: {len(raw_data)}")
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    print(f"\nBaseline Hofstede ΔH ({REGION_MAINLAND} - {REGION_HONGKONG}):")
    print(json.dumps(DELTA_HOFSTEDE, indent=2, ensure_ascii=False))

    delta_llm_data = calculate_delta_llm_with_se(raw_data)

    if not delta_llm_data:
        print("Error: no model deltas were computed. Check your data.")
        return

    full_results = {}
    for model_name, (delta_l, se_delta_l) in delta_llm_data.items():
        print(f"\n--- Analyzing model: {model_name} ---")
        print(f"LLM delta ΔL: {delta_l}")

        tier1_sensitivity = {
            f"tau_h_q{int(q * 100)}": analysis_1_directional_agreement(delta_l, se_delta_l, DELTA_HOFSTEDE, tau_h_quantile=q)
            for q in [0, 0.25, 0.40]
        }

        model_results = {
            "tier_1_directional_agreement_sensitivity": tier1_sensitivity,
            "tier_2_rank_order_agreement": analysis_2_rank_order_agreement(delta_l, DELTA_HOFSTEDE),
            "tier_3_affine_alignment": analysis_3_affine_alignment(delta_l, DELTA_HOFSTEDE),
            "tier_4_profile_similarity": analysis_4_profile_similarity(delta_l, DELTA_HOFSTEDE),
        }
        full_results[model_name] = model_results
        print(f"Model {model_name} done.")

    output_filepath = 'final_analysis_results_robust_REAL_DATA.json'
    with open(output_filepath, 'w', encoding='utf-8') as f:
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, np.bool_):
                    return bool(obj)
                return super().default(obj)

        json.dump(full_results, f, indent=4, ensure_ascii=False, cls=NpEncoder)

    print("\n--- Analysis complete ---")
    print(f"Saved detailed results to: {output_filepath}")


if __name__ == "__main__":
    main()

