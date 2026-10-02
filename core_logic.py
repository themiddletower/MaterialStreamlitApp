"""Inference and inverse design for CNT and carbon fiber."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from materials import ROOT, PROPERTIES, material_spec


def apply_fe(pct_arr, is_stat_arr, is_uv_arr, version):
    base = np.column_stack([pct_arr, is_stat_arr, is_uv_arr])
    c = np.asarray(pct_arr).reshape(-1, 1)
    if version == 'base':
        return base
    if version == 'log_only':
        return np.hstack([base, np.log(c + .1)])
    if version == 'inv_only':
        return np.hstack([base, 1 / (c + .1)**2])
    if version == 'full':
        # Retain the deployed transform exactly for the original CNT weights.
        return np.hstack([base, np.log(c + .1), 1 / (c + .001)**2, np.sin(c)])
    raise ValueError(f'Неизвестное преобразование признаков: {version}')


def load_hybrid_system(material_id='cnt'):
    spec = material_spec(material_id)
    if material_id == 'fiber':
        from fiber_models import load_fiber
        return load_fiber()
    import tensorflow as tf
    import gpflow
    base = ROOT / 'model_package'
    meta = joblib.load(base / 'metadata.pkl')
    meta.update(sk_indices=[0, 1], gp_indices=[3, 4, 2], method_labels=spec['methods'])
    sy = joblib.load(base / 'scaler_y.pkl')
    sx = joblib.load(base / 'scaler_x_gp.pkl')
    models = {k: joblib.load(base / f'sk_models/gpr_model_{k}.pkl') for k in meta['sk_indices']}
    scalers = {k: joblib.load(base / f'sk_models/scaler_x_{k}.pkl') for k in meta['sk_indices']}
    df = pd.read_excel(ROOT / 'raw_data_van.xlsx')
    method_columns = ['Статическое', 'УЗ+100', 'Статическое с смешением в УВ']
    df['method_code'] = df[method_columns].to_numpy().argmax(axis=1)
    grouped = df.groupby(['% УНТ/ 99% ПТФЭ', 'method_code'])[PROPERTIES].mean()
    y = grouped.to_numpy().copy()
    y[:, meta['log_indices']] = np.log1p(y[:, meta['log_indices']])
    ys = sy.transform(y)
    c, m = grouped.index.get_level_values(0).to_numpy(), grouped.index.get_level_values(1).to_numpy()
    xs = sx.transform(apply_fe(c, (m == 0).astype(float), (m == 2).astype(float), 'full'))
    xa = np.vstack([np.column_stack([xs, np.full(len(xs), k)]) for k in meta['gp_indices']])
    ya = np.vstack([ys[:, k:k+1] for k in meta['gp_indices']])
    kernel = gpflow.kernels.Matern52(lengthscales=[1.] * 6, active_dims=list(range(6)))
    kernel *= gpflow.kernels.Coregion(output_dim=5, rank=2, active_dims=[6])
    model = gpflow.models.GPR((xa, ya), kernel=kernel)
    latest = tf.train.latest_checkpoint(str(base / 'gpflow_weights'))
    if not latest:
        raise FileNotFoundError('Отсутствуют сохранённые веса GPflow для УНТ.')
    status = tf.train.Checkpoint(model=model).restore(latest)
    status.assert_existing_objects_matched()
    status.expect_partial()
    return dict(models_sk=models, scalers_x_sk=scalers, model_gp=model,
                scaler_x_gp=sx, sc_y=sy, meta=meta, material_id=material_id,
                training_means=grouped.to_numpy())


def predict_batch(concentrations, method_idx, bundle):
    spec = material_spec(bundle['material_id'])
    c = np.atleast_1d(np.asarray(concentrations, dtype=float))
    if c.ndim != 1 or not c.size or not np.isfinite(c).all():
        raise ValueError('Концентрация должна быть конечным числом.')
    low, high = spec['bounds']
    if np.any((c < low) | (c > high)):
        raise ValueError(f'Допустимая концентрация: {low:g}–{high:g}%.')
    if isinstance(method_idx, bool) or not isinstance(method_idx, (int, np.integer)) or not 0 <= method_idx < len(spec['methods']):
        raise ValueError('Недопустимый режим изготовления.')
    if bundle['material_id'] == 'fiber':
        from fiber_models import predict_fiber
        return predict_fiber(bundle, c, method_idx)
    meta, sy = bundle['meta'], bundle['sc_y']
    a, b = np.full(len(c), float(method_idx == 0)), np.full(len(c), float(method_idx == 2))
    mu, std = np.zeros((len(c), 5)), np.zeros((len(c), 5))
    for k in meta['sk_indices']:
        x = apply_fe(c, a, b, meta['sk_fe_versions'].get(k, 'base'))
        means, sigmas = bundle['models_sk'][k].predict(bundle['scalers_x_sk'][k].transform(x), return_std=True)
        mu[:, k], std[:, k] = means * sy.scale_[k] + sy.mean_[k], sigmas * sy.scale_[k]
    x = bundle['scaler_x_gp'].transform(apply_fe(c, a, b, 'full'))
    for k in meta['gp_indices']:
        means, var = bundle['model_gp'].predict_y(np.column_stack([x, np.full(len(c), k)]))
        mu[:, k] = means.numpy().ravel() * sy.scale_[k] + sy.mean_[k]
        std[:, k] = np.sqrt(np.maximum(var.numpy().ravel(), 0)) * sy.scale_[k]
    mu[:, meta['log_indices']] = np.expm1(mu[:, meta['log_indices']])
    return mu, std


def predict_hybrid(cnt, method_idx, bundle):
    """Central predictions; std stays in log1p space for log outputs."""
    mu, std = predict_batch([cnt], method_idx, bundle)
    return mu[0], std[0]


def prediction_intervals(predictions, std, bundle):
    mu, sigma = np.asarray(predictions), np.asarray(std)
    lower, upper = mu - 1.96 * sigma, mu + 1.96 * sigma
    for k in bundle['meta']['log_indices']:
        center = np.log1p(mu[..., k])
        lower[..., k] = np.expm1(center - 1.96 * sigma[..., k])
        upper[..., k] = np.expm1(center + 1.96 * sigma[..., k])
    return lower, upper


def scaled_targets(values, bundle):
    y = np.array(values, dtype=float, copy=True)
    y[..., bundle['meta']['log_indices']] = np.log1p(y[..., bundle['meta']['log_indices']])
    return bundle['sc_y'].transform(np.atleast_2d(y))


def solve_inverse_problem(target_dict, weights, bundle):
    props = bundle['meta']['prop_cols']
    if any(p not in target_dict for p in props):
        raise ValueError('Задайте все пять целевых свойств.')
    target = np.array([target_dict[p] for p in props], dtype=float)
    weights = np.asarray(weights, dtype=float)
    if not np.isfinite(target).all() or np.any(target < 0):
        raise ValueError('Цели должны быть конечными неотрицательными числами.')
    if weights.shape != (5,) or not np.isfinite(weights).all() or np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError('Нужны пять неотрицательных весов; хотя бы один должен быть положительным.')
    target_sc = scaled_targets(target, bundle)[0]
    spec = material_spec(bundle['material_id'])
    grid = np.linspace(*spec['bounds'], 101)
    best = (np.inf, float(grid[0]), 0)
    for method in range(len(spec['methods'])):
        def objective(c):
            values, _ = predict_hybrid(c, method, bundle)
            return float(np.sum(weights * (scaled_targets(values, bundle)[0] - target_sc)**2))
        values, _ = predict_batch(grid, method, bundle)
        losses = np.sum(weights * (scaled_targets(values, bundle) - target_sc)**2, axis=1)
        index = int(np.argmin(losses))
        candidates = [(float(losses[index]), float(grid[index])),
                      (float(losses[0]), float(grid[0])), (float(losses[-1]), float(grid[-1]))]
        # Refine each sampled local basin and include the exact endpoints.
        minima = np.flatnonzero((losses[1:-1] <= losses[:-2]) & (losses[1:-1] <= losses[2:])) + 1
        for i in minima:
            result = minimize_scalar(objective, bounds=(grid[i-1], grid[i+1]), method='bounded')
            if result.success and np.isfinite(result.fun):
                candidates.append((float(result.fun), float(result.x)))
        loss, concentration = min(candidates)
        best = min(best, (loss, concentration, method))
    _, concentration, method = best
    return concentration, method, predict_hybrid(concentration, method, bundle)[0]


def get_plot_data(prop_idx, bundle):
    if prop_idx not in range(5):
        raise ValueError('Неизвестное свойство.')
    grid = np.linspace(*material_spec(bundle['material_id'])['bounds'], 100)
    results = {}
    for m, label in enumerate(bundle['meta']['method_labels']):
        mu, std = predict_batch(grid, m, bundle)
        lower, upper = prediction_intervals(mu, std, bundle)
        results[label] = dict(x=grid, y=mu[:, prop_idx], lower=lower[:, prop_idx], upper=upper[:, prop_idx])
    return results


def experimental_means(material_id):
    material_spec(material_id)
    if material_id == 'fiber':
        return pd.read_csv(ROOT / 'data/fiber_experiments.csv').groupby(['concentration', 'method'])[PROPERTIES].mean().reset_index()
    df = pd.read_excel(ROOT / 'raw_data_van.xlsx')
    df['method'] = df.iloc[:, 1:4].to_numpy().argmax(axis=1)
    df['concentration'] = df.iloc[:, 0]
    return df.groupby(['concentration', 'method'])[PROPERTIES].mean().reset_index()
