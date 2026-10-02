import uuid

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core_logic import (
    load_hybrid_system, predict_hybrid, prediction_intervals,
    solve_inverse_problem, get_plot_data, experimental_means,
)
from materials import MATERIALS, PROPERTIES, DISPLAY_PROPERTIES, ROOT

st.set_page_config(page_title="Лаборатория композитов", layout="wide")


@st.cache_resource
def get_model_bundle(material_id):
    return load_hybrid_system(material_id)


@st.cache_data
def plot_data(material_id, property_index):
    return get_plot_data(property_index, get_model_bundle(material_id))


@st.cache_data
def measured_data(material_id):
    return experimental_means(material_id)


def make_figure(material_id, prop_idx):
    fig = go.Figure()
    observed = measured_data(material_id)
    for method, (label, data) in enumerate(plot_data(material_id, prop_idx).items()):
        color = ['#007f88', '#b24556', '#5a60b0'][method]
        fig.add_trace(go.Scatter(
            x=np.r_[data['x'], data['x'][::-1]],
            y=np.r_[data['upper'], data['lower'][::-1]], fill='toself',
            fillcolor=color, opacity=.12, line=dict(width=0),
            showlegend=False, hoverinfo='skip', legendgroup=label))
        fig.add_trace(go.Scatter(
            x=data['x'], y=data['y'], mode='lines', name=label,
            line=dict(color=color, width=2), legendgroup=label))
        points = observed[observed.method == method]
        fig.add_trace(go.Scatter(
            x=points.concentration, y=points[PROPERTIES[prop_idx]],
            mode='markers', marker=dict(color=color, size=7, symbol='diamond'),
            name=label + ' · эксперимент', showlegend=False, legendgroup=label))
    fig.update_layout(
        xaxis_title=f"Концентрация {MATERIALS[material_id]['short']}, %",
        yaxis_title=DISPLAY_PROPERTIES[prop_idx], hovermode='x unified',
        template='plotly_white', height=470,
        margin=dict(l=15, r=15, t=25, b=20),
        legend=dict(orientation='h', y=-.25, x=0),
    )
    return fig


def percent_difference(actual, target):
    return None if target == 0 else (actual - target) / target * 100


st.session_state.setdefault('results_by_material', {})
st.session_state.setdefault('optimizations_by_material', {})
st.session_state.setdefault('comparison_list', [])
st.session_state.setdefault('comparison_revision', 0)

with st.sidebar:
    st.subheader('Параметры состава')
    material_id = st.selectbox('Наполнитель', list(MATERIALS),
                              format_func=lambda key: MATERIALS[key]['label'], key='material')
    spec = MATERIALS[material_id]
    low, high = spec['bounds']
    concentration = st.number_input(
        f"Концентрация {spec['short']} (%)", min_value=low, max_value=high,
        value=spec['default'], step=.01, format='%.2f', key=f'concentration_{material_id}')
    with st.form(f'input_{material_id}'):
        method_idx = st.selectbox('Режим изготовления', range(len(spec['methods'])),
                                 format_func=lambda i: spec['methods'][i], key=f'method_{material_id}')
        submitted = st.form_submit_button('Рассчитать свойства', icon=':material/calculate:', width='stretch')
    st.caption(f"ПТФЭ + {spec['short']} · {low:g}–{high:g}%")

st.title('🧪 Лаборатория композитов')
st.caption(f"ПТФЭ / {spec['label']}")
try:
    bundle = get_model_bundle(material_id)
except (OSError, ValueError, RuntimeError) as error:
    st.error(f'Не удалось загрузить модели: {error}')
    st.stop()

if submitted or material_id not in st.session_state.results_by_material:
    p, s = predict_hybrid(concentration, method_idx, bundle)
    st.session_state.results_by_material[material_id] = dict(c=concentration, method=method_idx, p=p, s=s)

tab_predict, tab_optimize, tab_analytics = st.tabs(['🔮 Прогноз', '🎯 Оптимизация', '📈 Аналитика'])
with tab_predict:
    result = st.session_state.results_by_material[material_id]
    c, m, p, s = result['c'], result['method'], result['p'], result['s']
    lower, upper = prediction_intervals(p, s, bundle)
    st.subheader(f"{c:g}% {spec['short']} · {spec['methods'][m]}")
    view = st.radio('Представление', ['Показатели', 'Таблица'], horizontal=True,
                    key=f'view_{material_id}', label_visibility='collapsed')
    if view == 'Показатели':
        columns = st.columns(3)
        for i, label in enumerate(DISPLAY_PROPERTIES):
            with columns[i % 3]:
                st.metric(label, f'{p[i]:.3f}')
                st.caption(f'95% интервал: {lower[i]:.3f} … {upper[i]:.3f}')
    else:
        st.dataframe(pd.DataFrame({
            'Свойство': DISPLAY_PROPERTIES, 'Прогноз': p,
            'Нижняя граница 95%': lower, 'Верхняя граница 95%': upper,
        }), hide_index=True, width='stretch')
    if st.button('Добавить в сравнение', icon=':material/add:', key=f'add_{material_id}'):
        entry = dict(ID=uuid.uuid4().hex, Наполнитель=spec['label'],
                     Концентрация=c, Режим=spec['methods'][m])
        entry.update({prop: float(p[i]) for i, prop in enumerate(PROPERTIES)})
        st.session_state.comparison_list.append(entry)
        st.session_state.comparison_revision += 1
        st.toast('Состав добавлен')

with tab_optimize:
    st.subheader('Целевые свойства')
    with st.form(f'optimization_{material_id}'):
        targets, weights = {}, []
        default = np.mean(bundle['training_means'], axis=0)
        target_col, weight_col = st.columns(2)
        for i, (prop, label) in enumerate(zip(PROPERTIES, DISPLAY_PROPERTIES)):
            with target_col:
                targets[prop] = st.number_input(label, min_value=0., value=float(default[i]),
                                                format='%.3f', key=f'target_{material_id}_{i}')
            with weight_col:
                weights.append(st.slider('Вес · ' + label, 0., 5., 1., .1, key=f'weight_{material_id}_{i}'))
        optimize = st.form_submit_button('Подобрать состав', icon=':material/search:')
    if optimize:
        try:
            with st.spinner('Подбор состава'):
                best_c, best_m, best_p = solve_inverse_problem(targets, weights, bundle)
            st.session_state.optimizations_by_material[material_id] = dict(
                c=best_c, m=best_m, p=best_p, targets=targets.copy(), weights=list(weights))
        except ValueError as error:
            st.error(str(error))
    saved = st.session_state.optimizations_by_material.get(material_id)
    if saved:
        st.subheader(f"{saved['c']:.3f}% {spec['short']} · {spec['methods'][saved['m']]}")
        target_array = np.array([saved['targets'][prop] for prop in PROPERTIES])
        st.dataframe(pd.DataFrame({
            'Свойство': DISPLAY_PROPERTIES, 'Цель': target_array, 'Прогноз': saved['p'],
            'Отклонение': saved['p'] - target_array,
            'Отклонение, %': [percent_difference(float(pred), float(target))
                              for pred, target in zip(saved['p'], target_array)],
        }), hide_index=True, width='stretch')

with tab_analytics:
    selected = st.selectbox('Свойство', range(5), format_func=lambda i: DISPLAY_PROPERTIES[i],
                            key=f'property_{material_id}')
    if st.button('Построить график', icon=':material/show_chart:', key=f'plot_{material_id}'):
        st.session_state[f'plotted_{material_id}'] = selected
    plotted = st.session_state.get(f'plotted_{material_id}')
    if plotted is not None:
        st.subheader(DISPLAY_PROPERTIES[plotted])
        st.plotly_chart(make_figure(material_id, plotted), width='stretch')
    with st.expander('Экспериментальные средние'):
        observed = measured_data(material_id).copy()
        observed['method'] = observed.method.map(dict(enumerate(spec['methods'])))
        observed = observed.rename(columns={'concentration': 'Концентрация, %', 'method': 'Режим'})
        st.dataframe(observed, hide_index=True, width='stretch')
    if material_id == 'fiber':
        with st.expander('Проверка на исключённых концентрациях'):
            scores = pd.read_csv(ROOT / 'reports/fiber_cv.csv')
            chosen = [kind == bundle['selected'][PROPERTIES.index(prop)]
                      for kind, prop in zip(scores.model, scores.property)]
            scores = scores.loc[chosen, ['property', 'model', 'mae', 'r2', 'mape_percent']]
            st.dataframe(scores.rename(columns={'property': 'Свойство', 'model': 'Модель',
                         'mae': 'MAE', 'r2': 'R²', 'mape_percent': 'MAPE, %'}), hide_index=True, width='stretch')
            st.caption('Исключённые концентрации: 15%, 20%, 25%. Выбор моделей по тем же фолдам; независимой контрольной серии пока нет.')

if st.session_state.comparison_list:
    st.divider()
    st.subheader('Сравнение составов')
    comparison = pd.DataFrame(st.session_state.comparison_list)
    editor = comparison.copy()
    editor.insert(0, 'Удалить', False)
    edited = st.data_editor(
        editor, hide_index=True, width='stretch',
        disabled=comparison.columns.tolist(),
        column_config={'ID': None, 'Удалить': st.column_config.CheckboxColumn('Удалить', width='small'),
                       'Концентрация': st.column_config.NumberColumn('Концентрация, %', format='%.2f')},
        key=f'comparison_{st.session_state.comparison_revision}')
    delete_col, clear_col, export_col = st.columns(3)
    with delete_col:
        if st.button('Удалить выбранные', icon=':material/delete:'):
            remaining = edited.loc[~edited['Удалить'], comparison.columns].to_dict('records')
            st.session_state.comparison_list = remaining
            st.session_state.comparison_revision += 1
            st.rerun()
    with clear_col:
        if st.button('Очистить всё', icon=':material/clear_all:'):
            st.session_state.comparison_list = []
            st.session_state.comparison_revision += 1
            st.rerun()
    with export_col:
        st.download_button('Экспорт CSV', comparison.drop(columns='ID').to_csv(index=False).encode('utf-8-sig'),
                           'composite_comparison.csv', 'text/csv', icon=':material/download:')
