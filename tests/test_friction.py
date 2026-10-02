import json
import numpy as np
import pandas as pd
from core_logic import load_hybrid_system, predict_batch
from fiber_models import aggregate, predict_fiber
from friction_models import FRICTION_CONFIG, FRICTION_MODEL_NAME, engineered_features
from materials import ROOT
from tune_friction import evaluate


def test_feature_engineering_uses_concentration_and_two_indicators():
    x = np.array([[10., 1., 0.], [20., 0., 0.], [30., 0., 1.]])
    actual = engineered_features(x, 'full')
    np.testing.assert_allclose(actual[:, 0], x[:, 0])
    np.testing.assert_allclose(actual[:, 1], np.log(x[:, 0] + .1))
    np.testing.assert_allclose(actual[:, 2], 1 / x[:, 0]**2)
    np.testing.assert_allclose(actual[:, 3], np.sin(x[:, 0]))
    np.testing.assert_array_equal(actual[:, 4:], x[:, 1:])
    assert engineered_features(x, 'base', True).shape == (3, 5)


def test_friction_blocked_cv_is_reproducible():
    x, y, noise, _ = aggregate(pd.read_csv(ROOT / 'data/fiber_experiments.csv'))
    score, predictions = evaluate(FRICTION_CONFIG, x, y[:, 3], noise[:, 3])
    report = json.loads((ROOT / 'reports/friction_validation.json').read_text())
    assert np.isfinite(predictions).sum() == 9
    assert np.isnan(predictions[np.isin(x[:, 0], [10, 30])]).all()
    assert score['r2'] > .3
    for metric, value in score.items():
        np.testing.assert_allclose(value, report['interpolation'][metric], atol=1e-12)


def test_friction_is_smooth_and_other_fiber_predictions_unchanged():
    bundle = load_hybrid_system('fiber')
    assert bundle['selected'][3] == FRICTION_MODEL_NAME
    assert set(bundle['overrides']) == {3}
    original = dict(bundle, selected=bundle['base_selected'], overrides={})
    grid = np.linspace(10, 30, 401)
    for method in range(3):
        mean, std = predict_batch(grid, method, bundle)
        before, before_std = predict_fiber(original, grid, method)
        np.testing.assert_array_equal(mean[:, [0, 1, 2, 4]], before[:, [0, 1, 2, 4]])
        np.testing.assert_array_equal(std[:, [0, 1, 2, 4]], before_std[:, [0, 1, 2, 4]])
        assert np.max(np.abs(np.diff(mean[:, 3], n=2))) < 1e-12
        assert np.all(np.diff(mean[:, 3]) > 0)
        assert (std[:, 3] > 0).all()
