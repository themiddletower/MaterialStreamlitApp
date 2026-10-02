from streamlit.testing.v1 import AppTest
from materials import ROOT, PROPERTIES


def click_label(app, label):
    buttons = [b for b in app.button if b.label == label]
    assert len(buttons) == 1
    buttons[0].click().run()
    assert not app.exception
    assert app.title[0].value.startswith('🧪')
    assert [tab.label for tab in app.tabs] == ['🔮 Прогноз', '🎯 Оптимизация', '📈 Аналитика']


def test_switch_predict_plot_optimize_compare():
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=120).run()
    assert not app.exception
    assert len(app.metric) == 5
    click_label(app, 'Добавить в сравнение')
    app.selectbox(key='material').set_value('fiber').run()
    assert not app.exception
    assert app.number_input(key='concentration_fiber').min == 10
    assert app.number_input(key='concentration_fiber').max == 30
    assert 'активированное' in app.selectbox(key='method_fiber').options[2]
    app.number_input(key='concentration_fiber').set_value(22.5)
    app.selectbox(key='method_fiber').set_value(2)
    click_label(app, 'Рассчитать свойства')
    assert not any('R² < 0' in caption.value for caption in app.caption)
    click_label(app, 'Добавить в сравнение')
    comparison = app.session_state['comparison_list']
    assert len(comparison) == 2 and comparison[0]['Наполнитель'] != comparison[1]['Наполнитель']
    assert comparison[1]['Концентрация'] == 22.5
    revision = app.session_state['comparison_revision']
    app.session_state[f'comparison_{revision}'] = {
        'edited_rows': {0: {'Удалить': True}}, 'added_rows': [], 'deleted_rows': []}
    click_label(app, 'Удалить выбранные')
    assert len(app.session_state['comparison_list']) == 1
    assert app.session_state['comparison_list'][0]['Наполнитель'] == comparison[1]['Наполнитель']
    app.radio(key='view_fiber').set_value('Таблица').run()
    click_label(app, 'Построить график')
    assert len(app.get('plotly_chart')) == 1
    app.number_input(key='target_fiber_3').set_value(0)
    click_label(app, 'Подобрать состав')
    assert app.session_state['optimizations_by_material']['fiber']['targets'][PROPERTIES[3]] == 0
    app.selectbox(key='material').set_value('cnt').run()
    assert not app.exception
    assert 'ванна' in app.selectbox(key='method_cnt').options[2]
    assert app.number_input(key='concentration_cnt').max == 5
    assert len(app.get('plotly_chart')) == 0
    assert app.session_state['results_by_material']['cnt']['c'] == 2
    assert len(app.session_state['comparison_list']) == 1
    click_label(app, 'Очистить всё')
    assert app.session_state['comparison_list'] == []
