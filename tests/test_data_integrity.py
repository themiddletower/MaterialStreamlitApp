import hashlib
import shutil

import numpy as np
import pytest

import fiber_models
from data_integrity import data_digest, matches_training_data
from materials import ROOT


@pytest.mark.parametrize('newline', [b'\n', b'\r\n'])
def test_fingerprint_accepts_both_historical_representations(tmp_path, newline):
    lf = b'concentration,method,value\n10,0,0.35\n'
    path = tmp_path / 'experiments.csv'
    path.write_bytes(lf.replace(b'\n', newline))
    assert data_digest(path) == hashlib.sha256(lf).hexdigest()
    for historical in [lf, lf.replace(b'\n', b'\r\n')]:
        expected = hashlib.sha256(historical).hexdigest()
        assert matches_training_data(path, expected)
        path.write_bytes(lf.replace(b'0.35', b'0.36').replace(b'\n', newline))
        assert not matches_training_data(path, expected)
        path.write_bytes(lf.replace(b'\n', newline))


@pytest.mark.parametrize('newline', [b'\n', b'\r\n'])
@pytest.mark.parametrize('change_measurement', [False, True])
def test_real_model_loads_cross_platform_but_rejects_changed_data(
        tmp_path, monkeypatch, newline, change_measurement):
    original = fiber_models.load_fiber()
    data = (ROOT / 'data/fiber_experiments.csv').read_bytes().replace(b'\r\n', b'\n')
    if change_measurement:
        assert b'25.3' in data
        data = data.replace(b'25.3', b'25.4', 1)
    (tmp_path / 'data').mkdir()
    (tmp_path / 'data/fiber_experiments.csv').write_bytes(data.replace(b'\n', newline))
    model_folder = tmp_path / 'model_package/fiber'
    model_folder.mkdir(parents=True)
    shutil.copy2(ROOT / 'model_package/fiber/bundle.pkl', model_folder / 'bundle.pkl')
    monkeypatch.setattr(fiber_models, 'ROOT', tmp_path)
    if change_measurement:
        with pytest.raises(ValueError, match='Данные УВ изменены'):
            fiber_models.load_fiber()
    else:
        deployed = fiber_models.load_fiber()
        for method in range(3):
            expected = fiber_models.predict_fiber(original, [10, 17.5, 20, 30], method)
            actual = fiber_models.predict_fiber(deployed, [10, 17.5, 20, 30], method)
            np.testing.assert_array_equal(actual, expected)
