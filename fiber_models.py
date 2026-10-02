"""Independent and coregionalized GPRs with repeat-derived observation noise."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import joblib
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, DotProduct, RBF
from sklearn.preprocessing import StandardScaler

from materials import PROPERTIES, ROOT
from data_integrity import matches_training_data

LOG_INDICES = [2, 4]


def features(concentrations, methods):
    c = np.asarray(concentrations, dtype=float)
    m = np.asarray(methods)
    return np.column_stack([c, m == 0, m == 2]).astype(float)


def aggregate(df):
    groups = df.groupby(['concentration', 'method'], sort=True)[PROPERTIES]
    means = groups.mean()
    raw = means.to_numpy()
    y = raw.copy()
    sem = groups.std().to_numpy() / np.sqrt(groups.count().to_numpy())
    for k in LOG_INDICES:
        y[:, k] = np.log1p(raw[:, k])
        sem[:, k] /= 1 + raw[:, k]
    x = features(means.index.get_level_values(0), means.index.get_level_values(1))
    return x, y, np.maximum(sem**2, 1e-12), raw


def make_multi(x, y, noise):
    import gpflow
    gpflow.config.set_default_float(np.float64)
    n = x.shape[1]
    smooth = gpflow.kernels.Matern52(lengthscales=np.ones(n), active_dims=list(range(n)))
    coreg = gpflow.kernels.Coregion(output_dim=5, rank=2, active_dims=[n])
    # A nonzero, deterministic start avoids a stationary point at W=0.
    coreg.W.assign(np.random.default_rng(42).normal(0, 0.1, (5, 2)))
    xa = np.vstack([np.column_stack([x, np.full(len(x), k)]) for k in range(5)])
    ya = np.vstack([y[:, k:k+1] for k in range(5)])
    model = gpflow.models.GPR((xa, ya), kernel=smooth * coreg)
    model.likelihood.variance.assign(max(float(np.mean(noise)), 1e-5))
    gpflow.set_trainable(model.likelihood.variance, False)
    return model


def fit_candidate(x, y, noise, kind):
    sx, sy = StandardScaler().fit(x), StandardScaler().fit(y)
    xs, ys, ns = sx.transform(x), sy.transform(y), noise / sy.var_
    bundle = dict(scaler_x=sx, sc_y=sy, kind=kind)
    if kind == 'multi':
        import gpflow
        model = make_multi(xs, ys, ns)
        result = gpflow.optimizers.Scipy().minimize(
            model.training_loss, model.trainable_variables, options={'maxiter': 400})
        bundle.update(model_gp=model, gp_x=xs, gp_y=ys, gp_noise=ns,
                      convergence={'success': bool(result.success), 'message': str(result.message)})
    else:
        models = []
        for k in range(5):
            smooth = (Matern(np.ones(x.shape[1]), length_scale_bounds=(0.05, 100), nu=2.5)
                      if kind == 'matern' else RBF(np.ones(x.shape[1]), (0.05, 100)))
            kernel = ConstantKernel(1, (0.01, 100)) * smooth + ConstantKernel(.5, (.001, 100)) * DotProduct(.1, (.001, 10))
            model = GaussianProcessRegressor(kernel=kernel, alpha=np.maximum(ns[:, k], 1e-8),
                                            n_restarts_optimizer=4, random_state=42)
            model.fit(xs, ys[:, k])
            models.append(model)
        bundle['models'] = models
    return bundle


def predict_transformed(bundle, x):
    xs = bundle['scaler_x'].transform(x)
    if bundle['kind'] == 'multi':
        # Latent-function uncertainty, consistent with sklearn return_std.
        xa = np.vstack([np.column_stack([xs, np.full(len(xs), k)]) for k in range(5)])
        mu, var = bundle['model_gp'].predict_f(xa)
        mu = mu.numpy().reshape(5, -1).T
        std = np.sqrt(np.maximum(var.numpy().reshape(5, -1).T, 0))
    else:
        results = [m.predict(xs, return_std=True) for m in bundle['models']]
        mu, std = np.column_stack([r[0] for r in results]), np.column_stack([r[1] for r in results])
    return bundle['sc_y'].inverse_transform(mu), std * bundle['sc_y'].scale_


def to_physical(mu):
    result = mu.copy()
    result[:, LOG_INDICES] = np.expm1(result[:, LOG_INDICES])
    return result


def load_fiber():
    from materials import MATERIALS
    folder = ROOT / 'model_package' / 'fiber'
    info = joblib.load(folder / 'bundle.pkl')
    if not matches_training_data(ROOT / 'data/fiber_experiments.csv', info['data_sha256']):
        raise ValueError('Данные УВ изменены после обучения. Выполните python train_fiber.py.')
    if info['multi'] is not None:
        import gpflow
        stored = info['multi']
        model = make_multi(stored['gp_x'], stored['gp_y'], stored['gp_noise'])
        gpflow.utilities.multiple_assign(model, stored['parameters'])
        stored['model_gp'] = model
    info['meta'] = dict(prop_cols=PROPERTIES, log_indices=LOG_INDICES,
                        method_labels=MATERIALS['fiber']['methods'],
                        sk_indices=[k for k,v in enumerate(info['selected']) if v != 'multi'],
                        gp_indices=[k for k,v in enumerate(info['selected']) if v == 'multi'])
    info['material_id'] = 'fiber'
    return info


def predict_fiber(bundle, concentrations, method):
    x = features(concentrations, np.full(len(concentrations), method))
    mu, std = np.zeros((len(x), 5)), np.zeros((len(x), 5))
    overrides = bundle.get('overrides', {})
    for kind in {kind for k, kind in enumerate(bundle['selected']) if k not in overrides}:
        candidate = bundle['multi'] if kind == 'multi' else bundle['independent'][kind]
        means, sigmas = predict_transformed(candidate, x)
        indices = [k for k,v in enumerate(bundle['selected']) if v == kind and k not in overrides]
        mu[:, indices], std[:, indices] = means[:, indices], sigmas[:, indices]
    if overrides:
        from friction_models import predict_scalar
        for index, scalar in overrides.items():
            mu[:, index], std[:, index] = predict_scalar(scalar, x)
    return to_physical(mu), std
