import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core_logic import (load_hybrid_system, predict_hybrid, predict_batch,
                        prediction_intervals, solve_inverse_problem, get_plot_data)
from materials import ROOT, MATERIALS, PROPERTIES
from fiber_models import aggregate


@pytest.fixture(scope='module', params=['cnt', 'fiber'])
def bundle(request):
    return load_hybrid_system(request.param)


def test_original_cnt_predictions():
    bundle = load_hybrid_system()
    baseline = json.loads((ROOT / 'baseline_cnt.json').read_text())
    for case in baseline:
        mu, _ = predict_hybrid(case['concentration'], case['method'], bundle)
        np.testing.assert_allclose(mu, case['predictions'], rtol=1e-8, atol=1e-8)


def test_prediction_grid_and_intervals(bundle):
    concentrations = np.linspace(*MATERIALS[bundle['material_id']]['bounds'], 41)
    for method in range(3):
        mu, std = predict_batch(concentrations, method, bundle)
        lower, upper = prediction_intervals(mu, std, bundle)
        assert mu.shape == std.shape == (41, 5)
        assert np.isfinite([mu, std, lower, upper]).all()
        assert (mu > 0).all()
        assert (std >= 0).all()
        assert (lower <= mu).all() and (mu <= upper).all()
        np.testing.assert_allclose(mu[20], predict_hybrid(concentrations[20], method, bundle)[0], rtol=1e-7)
        for k in [2, 4]:
            np.testing.assert_allclose(np.log1p(upper[:,k]) - np.log1p(lower[:,k]), 3.92 * std[:,k], atol=1e-8)


def test_input_validation(bundle):
    lo, hi = MATERIALS[bundle['material_id']]['bounds']
    for concentration, method in [(lo-1,0), (hi+1,0), (np.nan,0), (np.inf,0), (lo,-1), (lo,3), (lo,1.5)]:
        with pytest.raises(ValueError):
            predict_hybrid(concentration, method, bundle)
    target = dict(zip(PROPERTIES, bundle['training_means'].mean(axis=0)))
    for weights in [[0]*5, [-1]*5, [1], [np.nan]*5]:
        with pytest.raises(ValueError):
            solve_inverse_problem(target, weights, bundle)
    with pytest.raises(ValueError):
        solve_inverse_problem({}, [1]*5, bundle)


def test_inverse_recovery_and_endpoints(bundle):
    lo, hi = MATERIALS[bundle['material_id']]['bounds']
    for concentration, method in [(lo,0), ((lo+hi)/2 + .17,2), (hi,1)]:
        target = predict_hybrid(concentration, method, bundle)[0]
        c, m, prediction = solve_inverse_problem(dict(zip(PROPERTIES, target)), np.ones(5), bundle)
        assert lo <= c <= hi and m == method
        assert abs(c-concentration) < .002
        np.testing.assert_allclose(prediction, target, rtol=1e-4)


def test_chart_bounds(bundle):
    lo, hi = MATERIALS[bundle['material_id']]['bounds']
    chart = get_plot_data(4, bundle)
    assert list(chart) == MATERIALS[bundle['material_id']]['methods']
    for data in chart.values():
        assert data['x'][0] == lo and data['x'][-1] == hi
        assert len(data['y']) == 100


def test_fiber_data_and_validation_protocol():
    df = pd.read_csv(ROOT / 'data/fiber_experiments.csv')
    assert len(df) == 135
    assert df.groupby(['concentration','method']).size().eq(9).all()
    first = df.iloc[0]
    assert first[PROPERTIES[3]] == .35 and first[PROPERTIES[4]] == .665
    assert first[PROPERTIES[0]] == 25.3
    x, y, noise, raw = aggregate(df)
    assert x.shape == (15, 3) and raw.shape == (15, 5)
    assert (noise > 0).all()
    protocol = json.loads((ROOT / 'reports/fiber_training.json').read_text(encoding='utf-8'))
    assert len(protocol['folds']) == 3
    for fold in protocol['folds']:
        assert fold['test_concentration'] not in fold['train_concentrations']
        assert fold['test_groups'] == 3
        assert {10,30} <= set(fold['train_concentrations'])


def test_loading_outside_app_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert np.isfinite(predict_hybrid(20, 0, load_hybrid_system('fiber'))[0]).all()
