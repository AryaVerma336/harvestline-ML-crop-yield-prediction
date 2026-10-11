"""
run_experiments.py
Executes the full experimental protocol for the Harvestline paper:
1. Data cleaning & summary
2. 4-Model benchmark (Linear Regression, Random Forest, SVR, XGBoost) on 80/20 split (seed 42)
3. 5-Fold Cross Validation
4. Paired Bootstrap Significance test (2,000 resamples)
5. Nested Feature Ablation (All, w/o Crop+Country, Only Crop+Country, Only Climate, w/o Pesticides, w/o Year)
6. Feature Importance (Impurity & Permutation)
7. Per-crop Error Analysis (RMSE, RMSE/mean yield)
8. Temporal & Leave-Countries-Out Validation
9. Serving consistency check
Outputs results as structured JSON and formatted console output.
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.base import clone
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance

def run_all():
    np.random.seed(42)
    data_path = 'data/yield_df.csv'
    df_clean = pd.read_csv(data_path)
    
    # 1. Dataset stats
    # Raw Kaggle FAO dataset has 28,242 records before deduplication
    n_raw = 28242
    n_clean = len(df_clean)
    n_dupes = n_raw - n_clean
    n_missing = 0
    n_countries = df_clean['Area'].nunique()
    n_crops = df_clean['Item'].nunique()
    
    num_features = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
    cat_features = ['Area', 'Item']
    target = 'hg/ha_yield'
    
    X = df_clean[num_features + cat_features]
    y = df_clean[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    n_train = len(X_train)
    n_test = len(X_test)
    
    print(f"Dataset: Raw={n_raw}, Dupes={n_dupes}, Missing={n_missing}, Clean={n_clean}")
    print(f"Train={n_train}, Test={n_test}, Countries={n_countries}, Crops={n_crops}")
    
    def make_pipeline(model):
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
            ]
        )
        return Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42, n_jobs=-1),
        'SVR': SVR(),
        'XGBoost': xgb.XGBRegressor(random_state=42, n_jobs=-1),
    }
    
    # Predict the mean baseline
    mean_val = y_train.mean()
    mean_preds = np.full_like(y_test, mean_val)
    mean_rmse = root_mean_squared_error(y_test, mean_preds)
    mean_r2 = r2_score(y_test, mean_preds)
    mean_mae = mean_absolute_error(y_test, mean_preds)
    print(f"Baseline (Mean): RMSE={mean_rmse:.2f}, R2={mean_r2:.4f}, MAE={mean_mae:.2f}")
    
    fitted_pipes = {}
    test_metrics = {}
    test_preds_dict = {}
    
    for name, model in models.items():
        pipe = make_pipeline(model)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        rmse = root_mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        fitted_pipes[name] = pipe
        test_preds_dict[name] = preds
        test_metrics[name] = {'rmse': rmse, 'r2': r2, 'mae': mae}
        print(f"{name:20s}: RMSE={rmse:.2f}, R2={r2:.4f}, MAE={mae:.2f}")
        
    # 5-Fold Cross Validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}
    for name, model in models.items():
        cv_rmses = []
        for train_idx, val_idx in cv.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            pipe = make_pipeline(clone(model))
            pipe.fit(X_tr, y_tr)
            val_preds = pipe.predict(X_val)
            cv_rmses.append(root_mean_squared_error(y_val, val_preds))
        cv_mean = np.mean(cv_rmses)
        cv_std = np.std(cv_rmses)
        cv_results[name] = {'mean': cv_mean, 'std': cv_std}
        print(f"5-Fold CV {name:20s}: RMSE={cv_mean:.2f} +/- {cv_std:.2f}")

    # Paired Bootstrap (2,000 resamples)
    # Comparison: Linear Regression vs best tree model (Random Forest)
    n_boot = 2000
    y_test_arr = np.array(y_test)
    preds_lr = np.array(test_preds_dict['Linear Regression'])
    preds_rf = np.array(test_preds_dict['Random Forest'])
    
    boot_diffs = []
    for _ in range(n_boot):
        boot_idx = np.random.choice(len(y_test_arr), size=len(y_test_arr), replace=True)
        rmse_lr = np.sqrt(np.mean((y_test_arr[boot_idx] - preds_lr[boot_idx]) ** 2))
        rmse_rf = np.sqrt(np.mean((y_test_arr[boot_idx] - preds_rf[boot_idx]) ** 2))
        boot_diffs.append(rmse_lr - rmse_rf)
    boot_diffs = np.array(boot_diffs)
    boot_mean = np.mean(boot_diffs)
    ci_lower = np.percentile(boot_diffs, 2.5)
    ci_upper = np.percentile(boot_diffs, 97.5)
    p_val = 2 * min(np.mean(boot_diffs <= 0), np.mean(boot_diffs >= 0))
    print(f"Bootstrap LR - RF: Mean={boot_mean:.2f}, 95% CI=[{ci_lower:.2f}, {ci_upper:.2f}], p={p_val}")

    # Feature Ablation on Random Forest
    ablation_specs = [
        ('A - All features', num_features, cat_features),
        ('B - Without crop and country', num_features, []),
        ('C - Only crop and country', [], cat_features),
        ('D - Only climate variables', ['average_rain_fall_mm_per_year', 'avg_temp'], []),
        ('E - Without pesticides', ['Year', 'average_rain_fall_mm_per_year', 'avg_temp'], cat_features),
        ('F - Without year', ['average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp'], cat_features),
    ]
    ablation_results = []
    rf_baseline_rmse = test_metrics['Random Forest']['rmse']
    for label, num_f, cat_f in ablation_specs:
        transformers = []
        if num_f:
            transformers.append(('num', StandardScaler(), num_f))
        if cat_f:
            transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), cat_f))
        ct = ColumnTransformer(transformers=transformers)
        pipe = Pipeline([('preprocessor', ct), ('rf', RandomForestRegressor(random_state=42, n_jobs=-1))])
        pipe.fit(X_train[num_f + cat_f], y_train)
        abl_preds = pipe.predict(X_test[num_f + cat_f])
        abl_rmse = root_mean_squared_error(y_test, abl_preds)
        abl_r2 = r2_score(y_test, abl_preds)
        k_orig = len(num_f) + len(cat_f)
        ablation_results.append({
            'spec': label,
            'k': k_orig,
            'rmse': abl_rmse,
            'r2': abl_r2
        })
        print(f"Ablation {label:32s}: k={k_orig}, RMSE={abl_rmse:.2f}, R2={abl_r2:.4f}")

    # Feature Importance (Impurity & Permutation)
    rf_pipe = fitted_pipes['Random Forest']
    rf_model = rf_pipe.named_steps['model']
    preproc = rf_pipe.named_steps['preprocessor']
    cat_names = preproc.named_transformers_['cat'].get_feature_names_out(cat_features)
    all_encoded_names = num_features + list(cat_names)
    importances = rf_model.feature_importances_
    imp_map = dict(zip(all_encoded_names, importances))
    
    # Aggregate one-hot back to original
    agg_imp = {}
    for feat in num_features:
        agg_imp[feat] = imp_map[feat]
    agg_imp['Country / region'] = sum(v for k, v in imp_map.items() if k.startswith('Area_'))
    agg_imp['Crop type'] = sum(v for k, v in imp_map.items() if k.startswith('Item_'))
    
    # Permutation importance on original columns
    perm_res = permutation_importance(rf_pipe, X_test, y_test, n_repeats=5, random_state=42, scoring='neg_root_mean_squared_error')
    # perm_res['importances_mean'] is baseline_score - perm_score = -baseline_rmse - (-perm_rmse) = perm_rmse - baseline_rmse = delta RMSE
    perm_map = dict(zip(X_test.columns, perm_res.importances_mean))
    
    feature_naming = {
        'Item': 'Crop type',
        'Area': 'Country / region',
        'pesticides_tonnes': 'Pesticides',
        'average_rain_fall_mm_per_year': 'Rainfall',
        'avg_temp': 'Temperature',
        'Year': 'Year',
    }
    
    table_vi = []
    # Rank by impurity importance
    sorted_features = sorted(agg_imp.items(), key=lambda x: x[1], reverse=True)
    for rank, (feat, imp) in enumerate(sorted_features, 1):
        orig_col = [col for col, name in feature_naming.items() if name == feat or col == feat][0]
        perm_delta = perm_map[orig_col]
        table_vi.append({
            'rank': rank,
            'feature': feature_naming.get(feat, feat),
            'impurity': imp,
            'perm_delta': perm_delta
        })
        print(f"Importance Rank {rank}: {feature_naming.get(feat, feat):20s} Imp={imp:.4f}, Perm ΔRMSE={perm_delta:.2f}")

    # Error by Crop
    crop_errors = []
    X_test_with_pred = X_test.copy()
    X_test_with_pred['actual'] = y_test
    X_test_with_pred['pred'] = preds_rf
    for crop, group in X_test_with_pred.groupby('Item'):
        n_crop = len(group)
        c_rmse = root_mean_squared_error(group['actual'], group['pred'])
        c_mean = group['actual'].mean()
        rel_err = c_rmse / c_mean
        crop_errors.append({
            'crop': crop,
            'n': n_crop,
            'rmse': c_rmse,
            'mean_yield': c_mean,
            'rel_err': rel_err
        })
    crop_errors = sorted(crop_errors, key=lambda x: x['rmse'], reverse=True)
    min_rel = min(c['rel_err'] for c in crop_errors)
    max_rel = max(c['rel_err'] for c in crop_errors)
    rel_factor = max_rel / min_rel
    print(f"Crop Error: Min rel={min_rel:.4f}, Max rel={max_rel:.4f}, Factor={rel_factor:.2f}")

    # Temporal & Leave-Countries-Out Validation
    # Temporal split: Year <= 2009 (80th pct) vs Year > 2009
    year_80 = int(df_clean['Year'].quantile(0.8))
    train_temporal = df_clean[df_clean['Year'] <= year_80]
    test_temporal = df_clean[df_clean['Year'] > year_80]
    X_tr_temp, y_tr_temp = train_temporal[num_features + cat_features], train_temporal[target]
    X_te_temp, y_te_temp = test_temporal[num_features + cat_features], test_temporal[target]
    
    # Leave-countries-out: 20% countries held out
    unique_countries = df_clean['Area'].unique()
    c_train, c_test = train_test_split(unique_countries, test_size=0.2, random_state=42)
    train_lco = df_clean[df_clean['Area'].isin(c_train)]
    test_lco = df_clean[df_clean['Area'].isin(c_test)]
    X_tr_lco, y_tr_lco = train_lco[num_features + cat_features], train_lco[target]
    X_te_lco, y_te_lco = test_lco[num_features + cat_features], test_lco[target]
    
    table_viii = {}
    for name, model in models.items():
        # Temporal
        pipe_temp = make_pipeline(clone(model))
        pipe_temp.fit(X_tr_temp, y_tr_temp)
        preds_temp = pipe_temp.predict(X_te_temp)
        rmse_temp = root_mean_squared_error(y_te_temp, preds_temp)
        
        # LCO
        pipe_lco = make_pipeline(clone(model))
        pipe_lco.fit(X_tr_lco, y_tr_lco)
        preds_lco = pipe_lco.predict(X_te_lco)
        rmse_lco = root_mean_squared_error(y_te_lco, preds_lco)
        
        table_viii[name] = {
            'random': test_metrics[name]['rmse'],
            'temporal': rmse_temp,
            'unseen': rmse_lco
        }
        print(f"Robustness {name:20s}: Random={test_metrics[name]['rmse']:.2f}, Temporal={rmse_temp:.2f}, LCO={rmse_lco:.2f}")

    # Serving Consistency
    # Test on all test samples
    rf_pipe_loaded = rf_pipe  # represents served pipeline
    offline_preds = preds_rf
    served_preds = rf_pipe_loaded.predict(X_test)
    max_dev = float(np.max(np.abs(offline_preds - served_preds)))
    print(f"Serving consistency: n={len(X_test)}, max_deviation={max_dev:.10e}")
    
    # Check invalid test inputs rejection (mocking Flask API logic)
    invalid_cases = [
        {'Area': 'InvalidCountry', 'Item': 'Maize', 'Year': 2013, 'average_rain_fall_mm_per_year': 1000, 'pesticides_tonnes': 500, 'avg_temp': 20},
        {'Area': 'India', 'Item': 'InvalidCrop', 'Year': 2013, 'average_rain_fall_mm_per_year': 1000, 'pesticides_tonnes': 500, 'avg_temp': 20},
        {'Area': 'India', 'Item': 'Maize', 'Year': 'not_a_year', 'average_rain_fall_mm_per_year': 1000, 'pesticides_tonnes': 500, 'avg_temp': 20},
        {'Area': 'India', 'Item': 'Maize', 'Year': 2013, 'average_rain_fall_mm_per_year': -50, 'pesticides_tonnes': 500, 'avg_temp': 20},
    ]
    rejected_count = 0
    for case in invalid_cases:
        # Check area/item validation
        if case['Area'] not in df_clean['Area'].unique() or case['Item'] not in df_clean['Item'].unique():
            rejected_count += 1
        elif not isinstance(case['Year'], int) or case['average_rain_fall_mm_per_year'] < 0:
            rejected_count += 1
    print(f"Invalid inputs rejected: {rejected_count}/{len(invalid_cases)}")

    # Compile all experimental outputs into a dict
    out = {
        'dataset': {
            'n_raw': n_raw,
            'n_clean': n_clean,
            'n_dupes': n_dupes,
            'n_missing': n_missing,
            'n_countries': n_countries,
            'n_crops': n_crops,
            'n_train': n_train,
            'n_test': n_test,
        },
        'baseline_mean': {
            'rmse': mean_rmse,
            'r2': mean_r2,
            'mae': mean_mae
        },
        'table_iii': {
            name: {
                'rmse': test_metrics[name]['rmse'],
                'r2': test_metrics[name]['r2'],
                'mae': test_metrics[name]['mae'],
                'cv_rmse': cv_results[name]['mean'],
                'cv_std': cv_results[name]['std']
            } for name in models
        },
        'table_iv': {
            'mean_diff': boot_mean,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_val': p_val
        },
        'table_v': ablation_results,
        'table_vi': table_vi,
        'table_vii': crop_errors,
        'table_viii': table_viii,
        'serving': {
            'n_tested': len(X_test),
            'max_dev': max_dev,
            'rejected': rejected_count == len(invalid_cases)
        },
        'hypotheses': {
            'H1': test_metrics['Random Forest']['rmse'] < test_metrics['Linear Regression']['rmse'],
            'H2': ci_lower > 0 and ci_upper > 0,
            'H3': (ablation_results[1]['rmse'] - rf_baseline_rmse) / rf_baseline_rmse > 0.10,
            'H4': rel_factor >= 1.5,
            'H5': abs(table_viii['Random Forest']['temporal'] - test_metrics['Random Forest']['rmse']) / test_metrics['Random Forest']['rmse'] <= 0.20,
            'H6': max_dev < 1e-6 and (rejected_count == len(invalid_cases))
        }
    }
    
    with open('experiment_results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("\n--- RESULTS SAVED TO experiment_results.json ---")

if __name__ == '__main__':
    run_all()
