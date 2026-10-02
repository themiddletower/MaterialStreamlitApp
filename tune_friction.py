"""Reproducible candidate search on the unchanged concentration-blocked folds."""
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import r2_score, mean_absolute_error
from fiber_models import aggregate
from friction_models import fit_scalar, predict_scalar
from materials import ROOT


def candidates():
    for features in ['base', 'log', 'log_only', 'full', 'smooth', 'quadratic', 'cubic']:
        for interactions in [False, True]:
            yield dict(family='bayesian', features=features, interactions=interactions)
            for alpha in [.01, .1, 1., 10., 100.]:
                yield dict(family='ridge', features=features, interactions=interactions, alpha=alpha)
    for features in ['base', 'log', 'log_only', 'full', 'smooth']:
        for family in ['rbf', 'matern']:
            for floor in [.3, .7, 1.5]:
                for noise in [.01, .1]:
                    yield dict(family=family, features=features, length_floor=floor, noise_floor=noise)


def evaluate(config, x, y, noise):
    mask = np.isin(x[:, 0], [15., 20., 25.])
    prediction = np.full(len(y), np.nan)
    for concentration in [15., 20., 25.]:
        test = x[:, 0] == concentration
        fitted = fit_scalar(x[~test], y[~test], noise[~test], config)
        prediction[test] = predict_scalar(fitted, x[test])[0]
    return dict(r2=float(r2_score(y[mask], prediction[mask])),
                mae=float(mean_absolute_error(y[mask], prediction[mask])),
                mape_percent=float(np.mean(np.abs((prediction[mask] - y[mask]) / y[mask])) * 100)), prediction


def main():
    x, y, noise, _ = aggregate(pd.read_csv(ROOT / 'data/fiber_experiments.csv'))
    results, predictions = [], []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        for index, config in enumerate(candidates()):
            score, prediction = evaluate(config, x, y[:, 3], noise[:, 3])
            row = dict(candidate=index, **score, **config)
            results.append(row)
            for i in np.flatnonzero(np.isfinite(prediction)):
                predictions.append(dict(candidate=index, concentration=x[i, 0],
                                        method=0 if x[i, 1] else 2 if x[i, 2] else 1,
                                        actual=y[i, 3], prediction=prediction[i]))
            if index % 10 == 0:
                print(index, 'best R2', max(r['r2'] for r in results), flush=True)
    frame = pd.DataFrame(results).sort_values('r2', ascending=False)
    frame.to_csv(ROOT / 'reports/friction_search.csv', index=False)
    pd.DataFrame(predictions).to_csv(ROOT / 'reports/friction_search_predictions.csv', index=False)
    winner = max(results, key=lambda row: row['r2'])
    (ROOT / 'reports/friction_search_best.json').write_text(json.dumps(winner, indent=2), encoding='utf-8')
    print(frame.head(20).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
