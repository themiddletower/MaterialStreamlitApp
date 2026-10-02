from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROPERTIES = [
    'Предел прочности при разрыве, МПа',
    'Относительное удлинение при разрыве, %',
    'Модуль упругости, МПа',
    'Коэффициент трения',
    'Износостойкость (скорость потери массы мг/час)',
]
DISPLAY_PROPERTIES = [
    'Прочность, МПа', 'Удлинение, %', 'Модуль упругости, МПа',
    'Коэффициент трения', 'Скорость износа, мг/ч',
]
MATERIALS = {
    'cnt': {
        'label': 'Углеродные нанотрубки (УНТ)', 'short': 'УНТ',
        'bounds': (1.0, 5.0), 'default': 2.0,
        'methods': ['Статическое', 'УЗ+100', 'Статическое + ультразвуковая ванна'],
    },
    'fiber': {
        'label': 'Углеродное волокно (УВ)', 'short': 'УВ',
        'bounds': (10.0, 30.0), 'default': 20.0,
        'methods': ['Статическое', 'УЗ+100', 'Статическое + активированное волокно'],
    },
}


def material_spec(material_id):
    if material_id not in MATERIALS:
        raise ValueError(f'Неизвестный наполнитель: {material_id}')
    return MATERIALS[material_id]
