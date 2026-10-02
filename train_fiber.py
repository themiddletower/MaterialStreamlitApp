"""Train once, compare concentration-blocked interpolation folds, save artifacts."""
import hashlib
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, r2_score

from fiber_models import aggregate, fit_candidate, predict_transformed, to_physical
from materials import PROPERTIES, ROOT


def main():
    data_path = ROOT / 'data/fiber_experiments.csv'
    data = pd.read_csv(data_path)
    x, y, noise, raw = aggregate(data)
    kinds = ['matern', 'rbf', 'multi']
    concentrations = sorted(data.concentration.unique())
    internal = concentrations[1:-1]
    predictions = {kind: np.full_like(raw, np.nan) for kind in kinds}
    folds, convergence = [], []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        for concentration in internal:
            test = x[:, 0] == concentration
            train = ~test
            folds.append(dict(test_concentration=concentration, train_groups=int(train.sum()),
                              test_groups=int(test.sum()), train_concentrations=x[train, 0].tolist()))
            for kind in kinds:
                print(f'CV holdout={concentration:g}%, model={kind}', flush=True)
                bundle = fit_candidate(x[train], y[train], noise[train], kind)
                mu, _ = predict_transformed(bundle, x[test])
                predictions[kind][test] = to_physical(mu)
                if 'convergence' in bundle:
                    convergence.append(dict(fold=float(concentration), **bundle['convergence']))
        mask = np.isin(x[:, 0], internal)
        scores, selected = [], []
        for k, prop in enumerate(PROPERTIES):
            candidates = []
            for kind in kinds:
                yp, yt = predictions[kind][mask, k], raw[mask, k]
                row = dict(property=prop, model=kind, mae=float(mean_absolute_error(yt, yp)),
                           r2=float(r2_score(yt, yp)), mape_percent=float(np.mean(np.abs((yp-yt)/yt))*100))
                scores.append(row)
                candidates.append(row)
            selected.append(min(candidates, key=lambda row: row['mae'])['model'])
        final = dict(selected=selected, independent={}, multi=None,
                     data_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
                     sklearn_version=sklearn.__version__)
        for kind in sorted(set(selected)):
            print('FINAL', kind, flush=True)
            bundle = fit_candidate(x, y, noise, kind)
            if kind == 'multi':
                import gpflow
                bundle['parameters'] = {str(k): v.numpy() for k,v in gpflow.utilities.parameter_dict(bundle['model_gp']).items()}
                del bundle['model_gp']
                final['multi'] = bundle
            else:
                final['independent'][kind] = bundle
        # A common target scaler also defines the inverse-design objective.
        from sklearn.preprocessing import StandardScaler
        final['sc_y'] = StandardScaler().fit(y)
        final['training_means'] = raw
        warning_messages = sorted(set(str(w.message) for w in caught))
    folder = ROOT / 'model_package/fiber'
    folder.mkdir(exist_ok=True)
    joblib.dump(final, folder / 'bundle.pkl')
    report = ROOT / 'reports'
    report.mkdir(exist_ok=True)
    frame = pd.DataFrame(scores)
    frame.to_csv(report / 'fiber_cv.csv', index=False, encoding='utf-8-sig')
    details = []
    for kind in kinds:
        for i in np.flatnonzero(mask):
            for k, prop in enumerate(PROPERTIES):
                details.append(dict(model=kind, concentration=x[i,0], method=int(data.groupby(['concentration','method']).size().index[i][1]),
                                    property=prop, actual=raw[i,k], prediction=predictions[kind][i,k]))
    pd.DataFrame(details).to_csv(report / 'fiber_cv_predictions.csv', index=False, encoding='utf-8-sig')
    (report / 'fiber_training.json').write_text(json.dumps(dict(
        folds=folds, selected=selected, convergence=convergence, warnings=warning_messages,
        seed=42, source_sha256=final['data_sha256'], sklearn_version=sklearn.__version__,
        protocol='Leave one internal concentration out across all methods; 10 and 30 always train; scalers fit inside each fold; score group means; select by MAE; not independent final-test metrics.'
    ), ensure_ascii=False, indent=2), encoding='utf-8')
    text = [
        '# Базовое сравнение моделей углеродного волокна', '',
        'Это первоначальное сравнение GPR. Финальная модель трения уточняется отдельно; '
        'актуальный результат и ограничения оценки: [friction_validation.md](friction_validation.md). '
        'Другие четыре свойства не меняются.', '',
        '135 повторных измерений, 15 сочетаний концентрации и режима. '
        'Проверка: поочерёдно исключены 15%, 20%, 25% целиком со всеми режимами; '
        'границы 10% и 30% всегда в обучении. Скейлеры обучены внутри каждого фолда. '
        'Метрики рассчитаны по девяти исключённым групповым средним, а не по обучению.', '',
        'Сравниваются независимые Matérn 5/2 + линейное ядро, RBF + линейное ядро '
        'и совместный GPflow Matérn 5/2 × Coregion(rank=2). '
        'Независимые модели используют дисперсию среднего s²/n для каждой точки; '
        'GPflow использует фиксированный общий шум, усреднённый в нормированной шкале. '
        'Модуль и износ преобразованы через log1p; ошибки рассчитаны в физических единицах.', '',
        '| Свойство | Модель | MAE | R² | MAPE, % | Выбрана |',
        '| --- | --- | ---: | ---: | ---: | --- |',
    ]
    for row in scores:
        k = PROPERTIES.index(row['property'])
        text.append(f"| {row['property']} | {row['model']} | {row['mae']:.5f} | {row['r2']:.4f} | {row['mape_percent']:.2f} | {'да' if selected[k] == row['model'] else ''} |")
    text += ['', 'Выбор выполнен по минимальной MAE для каждого свойства. Эти же фолды '
             'использовались для выбора, поэтому оценки не являются независимой проверкой '
             'готового гибрида. Для неё нужны новые концентрации. После выбора модели '
             'обучены на всех 15 условиях. УНТ не переобучались.', '',
             'Интервалы характеризуют неопределённость регрессии, а не полный разброс '
             'единичного лабораторного испытания. При логарифмировании центральный прогноз '
             'является медианой обратного распределения. Предсказание износа: меньше = лучше.', '',
             'В исходном листе единица удлинения ошибочно подписана «МПа». '
             'Сохранены исходные числа; для одноимённого показателя используются проценты. '
             'Лист чистого ПТФЭ и дополнительные точки УНТ не добавлялись к сохранённым моделям.', '',
             'Параметры оптимизации, предупреждения и результаты по каждому фолду: '
             '`fiber_training.json` и `fiber_cv_predictions.csv`.']
    (report / 'fiber_validation.md').write_text('\n'.join(text), encoding='utf-8')
    print(frame.to_string(index=False), flush=True)
    print('Selected:', selected, flush=True)
    from refine_friction import install
    install()


if __name__ == '__main__':
    main()
