"""Smooth scalar regressors with concentration-only feature engineering."""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, WhiteKernel
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.preprocessing import StandardScaler

FRICTION_CONFIG = dict(family='ridge', features='base', interactions=True, alpha=.01)
FRICTION_MODEL_NAME = 'ridge_interactions'


def engineered_features(x, version='base', interactions=False):
    x = np.asarray(x, dtype=float)
    c = x[:, :1]
    modes = x[:, 1:3]
    if version == 'base':
        continuous = c
    elif version == 'log':
        continuous = np.hstack([c, np.log(c + .1)])
    elif version == 'log_only':
        continuous = np.log(c + .1)
    elif version == 'full':
        continuous = np.hstack([c, np.log(c + .1), 1 / c**2, np.sin(c)])
    elif version == 'smooth':
        continuous = np.hstack([c, np.log(c + .1), 1 / c**2])
    elif version == 'quadratic':
        continuous = np.hstack([c, c**2])
    elif version == 'cubic':
        continuous = np.hstack([c, c**2, c**3])
    else:
        raise ValueError(f'Unknown feature version: {version}')
    result = [continuous, modes]
    if interactions:
        result.extend([continuous * modes[:, i:i+1] for i in range(2)])
    return np.hstack(result)


def fit_scalar(x, y, noise, config):
    design = engineered_features(x, config['features'], config.get('interactions', False))
    sx, sy = StandardScaler().fit(design), StandardScaler().fit(y[:, None])
    xs, ys = sx.transform(design), sy.transform(y[:, None]).ravel()
    family = config['family']
    if family == 'ridge':
        model = Ridge(alpha=config['alpha']).fit(xs, ys)
        # Posterior-style uncertainty for ridge, with residual degrees of freedom.
        xa = np.column_stack([np.ones(len(xs)), xs])
        penalty = np.diag([0.] + [config['alpha']] * xs.shape[1])
        covariance = np.linalg.pinv(xa.T @ xa + penalty)
        dof = max(len(xs) - np.trace(xa @ covariance @ xa.T), 1.)
        variance = float(np.sum((ys - model.predict(xs))**2) / dof)
        extra = dict(covariance=covariance * variance)
    elif family == 'bayesian':
        model = BayesianRidge().fit(xs, ys)
        extra = {}
    else:
        floor = config['length_floor']
        lengths = np.ones(xs.shape[1]) if config.get('ard', True) else 1.
        smooth = (RBF(lengths, (floor, 100)) if family == 'rbf'
                  else Matern(lengths, (floor, 100), nu=2.5))
        kernel = ConstantKernel(1, (.01, 100)) * smooth
        kernel += WhiteKernel(config['noise_floor'], (config['noise_floor'], 1.))
        model = GaussianProcessRegressor(
            kernel=kernel, alpha=np.maximum(noise / sy.var_[0], 1e-8),
            random_state=42, n_restarts_optimizer=1).fit(xs, ys)
        extra = {}
    return dict(model=model, scaler_x=sx, scaler_y=sy, config=config, **extra)


def predict_scalar(bundle, x):
    cfg = bundle['config']
    design = engineered_features(x, cfg['features'], cfg.get('interactions', False))
    xs = bundle['scaler_x'].transform(design)
    if cfg['family'] == 'ridge':
        mu = bundle['model'].predict(xs)
        xa = np.column_stack([np.ones(len(xs)), xs])
        var = np.einsum('ij,jk,ik->i', xa, bundle['covariance'], xa)
    else:
        mu, std = bundle['model'].predict(xs, return_std=True)
        var = std**2
        if cfg['family'] == 'bayesian':
            var -= 1 / bundle['model'].alpha_
        else:
            # Remove WhiteKernel observation noise: intervals describe the latent curve.
            var -= bundle['model'].kernel_.k2.noise_level
    sy = bundle['scaler_y']
    return mu * sy.scale_[0] + sy.mean_[0], np.sqrt(np.maximum(var, 0)) * sy.scale_[0]
