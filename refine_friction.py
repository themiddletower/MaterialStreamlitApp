"""Install the selected smooth friction model without changing other properties."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from fiber_models import aggregate
from friction_models import FRICTION_CONFIG, FRICTION_MODEL_NAME, fit_scalar, predict_scalar
from materials import ROOT, PROPERTIES
from tune_friction import evaluate
from data_integrity import matches_training_data


def install():
    data_path = ROOT / 'data/fiber_experiments.csv'
    package_path = ROOT / 'model_package/fiber/bundle.pkl'
    bundle = joblib.load(package_path)
    if not matches_training_data(data_path, bundle['data_sha256']):
        raise ValueError('First retrain all fiber models with train_fiber.py.')
    x, y, noise, _ = aggregate(pd.read_csv(data_path))
    score, predictions = evaluate(FRICTION_CONFIG, x, y[:, 3], noise[:, 3])
    scalar = fit_scalar(x, y[:, 3], noise[:, 3], FRICTION_CONFIG)
    bundle.setdefault('base_selected', list(bundle['selected']))
    bundle.setdefault('overrides', {})[3] = scalar
    bundle['selected'][3] = FRICTION_MODEL_NAME
    joblib.dump(bundle, package_path)

    reports = ROOT / 'reports'
    cv = pd.read_csv(reports / 'fiber_cv.csv')
    cv = cv[cv.model != FRICTION_MODEL_NAME]
    cv = pd.concat([cv, pd.DataFrame([dict(property=PROPERTIES[3], model=FRICTION_MODEL_NAME, **score)])], ignore_index=True)
    cv.to_csv(reports / 'fiber_cv.csv', index=False, encoding='utf-8-sig')
    details = pd.read_csv(reports / 'fiber_cv_predictions.csv')
    details = details[details.model != FRICTION_MODEL_NAME]
    rows = [dict(model=FRICTION_MODEL_NAME, concentration=x[i, 0],
                 method=0 if x[i, 1] else 2 if x[i, 2] else 1, property=PROPERTIES[3],
                 actual=y[i, 3], prediction=predictions[i])
            for i in np.flatnonzero(np.isfinite(predictions))]
    pd.concat([details, pd.DataFrame(rows)], ignore_index=True).to_csv(
        reports / 'fiber_cv_predictions.csv', index=False, encoding='utf-8-sig')
    protocol = json.loads((reports / 'fiber_training.json').read_text(encoding='utf-8'))
    protocol['base_selected'] = bundle['base_selected']
    protocol['selected'] = bundle['selected']
    protocol['friction_refinement'] = dict(config=FRICTION_CONFIG, metrics=score,
        protocol='Same 15/20/25 concentration folds; hyperparameter-selection scores, not independent test results.')
    (reports / 'fiber_training.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')

    # Boundary folds test extrapolation too; report separately from the original metric.
    all_predictions = np.zeros(len(y))
    for concentration in np.unique(x[:, 0]):
        test = x[:, 0] == concentration
        model = fit_scalar(x[~test], y[~test, 3], noise[~test, 3], FRICTION_CONFIG)
        all_predictions[test] = predict_scalar(model, x[test])[0]
    boundary_score = dict(r2=float(r2_score(y[:, 3], all_predictions)),
                          mae=float(mean_absolute_error(y[:, 3], all_predictions)))
    (reports / 'friction_validation.json').write_text(json.dumps(dict(
        config=FRICTION_CONFIG, interpolation=score, five_concentration_diagnostic=boundary_score,
        source_sha256=bundle['data_sha256'], actual=y[:, 3].tolist(),
        interpolation_predictions=[None if not np.isfinite(v) else float(v) for v in predictions],
        all_concentration_predictions=all_predictions.tolist()), indent=2), encoding='utf-8')
    search = pd.read_csv(reports / 'friction_search.csv')
    best_features = search.sort_values('r2', ascending=False).drop_duplicates('features')
    baseline = cv[(cv.model == 'matern') & (cv.property == PROPERTIES[3])].iloc[0]
    text = [
        '# Уточнение модели коэффициента трения УВ', '',
        'Использованы те же 135 измерений, 15 групп и три фолда: полностью исключается '
        'концентрация 15%, 20% или 25% во всех режимах. Границы 10% и 30% остаются в обучении. '
        'Скейлеры обучаются только на обучающей части. Метрики относятся к девяти исключённым '
        'средним, а не к повторным измерениям и не к обучению.', '',
        '| Вариант | R² | MAE | MAPE, % |', '| --- | ---: | ---: | ---: |',
        f"| Предыдущий Matérn GPR | {baseline.r2:.6f} | {baseline.mae:.6f} | {baseline.mape_percent:.3f} |",
        f"| Ridge с взаимодействиями режима и концентрации | {score['r2']:.6f} | {score['mae']:.6f} | {score['mape_percent']:.3f} |", '',
        'Выбрана регуляризованная модель y = b + a·c + d₀·I₀ + d₂·I₂ + e₀·c·I₀ + e₂·c·I₂. '
        'Режим УЗ+100 является опорным, поэтому лишнего one-hot столбца нет. '
        'После стандартизации используется Ridge(alpha=0.01). В каждом режиме кривая линейная, '
        'без узких пиков около обучающих концентраций. Это приближение общего тренда, '
        'а не точное воспроизведение каждого экспериментального среднего.', '',
        f'Перебраны {len(search)} конфигурации: Ridge и BayesianRidge, GPR с RBF/Matérn, '
        'границами длины корреляции 0.3/0.7/1.5 и нижними уровнями дополнительного шума 0.01/0.1 '
        'в нормированной шкале. Для Ridge проверены alpha = 0.01/0.1/1/10/100. '
        'Все преобразования применяются к концентрации, а не к показателям свойств.', '',
        '| Признаки | Лучший R² | Семейство |', '| --- | ---: | --- |',
    ]
    for row in best_features.itertuples():
        text.append(f'| {row.features} | {row.r2:.6f} | {row.family} |')
    text += ['',
        '`base`: c; `log`: c и log(c+0.1); `log_only`: только log(c+0.1); '
        '`full`: c, log(c+0.1), 1/c², sin(c); `smooth`: то же без sin; '
        '`quadratic`/`cubic`: полиномы второй/третьей степени. Во всех вариантах есть два '
        'индикатора режима; для линейных моделей отдельно проверены взаимодействия. '
        'Исходные значения концентрации выражены в процентах. Предложенное обогащение '
        'проверено, но для этих данных оказалось слабее простого регуляризованного тренда.', '',
        'Длины корреляции прежнего GPR могли сжиматься до нижней границы 0.05. '
        'В сочетании с малой дисперсией среднего это давало почти точную подгонку узлов '
        'и локальные изгибы между ними. Новая модель не интерполирует узлы принудительно.', '',
        '**Ограничения проверки.** Те же фолды использованы для выбора среди 144 конфигураций, '
        'поэтому R² = 0.365 не является независимой оценкой после подбора. '
        'Новые экспериментальные концентрации нужны для независимого подтверждения. '
        'Нельзя считать сглаживание доказательством линейного физического закона. '
        'Провал экспериментальных средних около 15% сглаживается, а не удаляется из данных.', '',
        f"Дополнительная диагностика с исключением каждой из пяти концентраций: R² = {boundary_score['r2']:.6f}, "
        f"MAE = {boundary_score['mae']:.6f}. В крайних фолдах это экстраполяция; результат нельзя "
        'напрямую смешивать с основными тремя интерполяционными фолдами. Это также не независимая серия.', '',
        '95% интервалы построены по приближённой ковариации коэффициентов Ridge с оценкой '
        'остаточной дисперсии и эффективного числа степеней свободы. Они описывают '
        'неопределённость среднего тренда, не полный разброс отдельных образцов.', '',
        'Другие четыре свойства УВ и все сохранённые модели УНТ не изменены. '
        'Воспроизведение: `python tune_friction.py`, затем `python refine_friction.py`. '
        'Полный `python train_fiber.py` также применяет выбранное уточнение после обучения.', '',
        'Все кандидаты: `friction_search.csv`; их исключённые прогнозы: '
        '`friction_search_predictions.csv`; итог: `friction_validation.json`.',
    ]
    (reports / 'friction_validation.md').write_text('\n'.join(text), encoding='utf-8')
    print('Friction installed:', score, flush=True)
    print('Five-concentration diagnostic:', boundary_score, flush=True)


if __name__ == '__main__':
    install()
