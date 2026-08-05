"""epi -- работа с записями ЭЭГ Nicolet/Nervus (.e) для задачи эпилепсии.

Пакет закрывает путь от папки с файлами до обучающей выборки:

    from epi import Catalog, WindowSet, summary, render

    catalog = Catalog("catalog.sqlite")
    catalog.sync("/путь/к/данным")     # 1. обойти папки и собрать метаданные
    catalog.patient(ключ)              # 2. данные одного пациента
    windows = WindowSet(catalog)       # 3. нарезать обучающую выборку
    windows.build(); windows.split()
    x, y = windows.sample("train", n=512)
    render(summary(catalog))           # 4. выжимка статистики

Из чего состоит пакет:

    reader    -- NicoletEReader: чтение сигнала, каналов, сегментов и маркеров
                 из одного файла .e;
    catalog   -- Catalog: каталог на SQLite, обход папок и поиск по метаданным;
    windows   -- WindowSet: нарезка на окна, разбиение по пациентам, выборка;
    stats     -- summary/render: сводная статистика по каталогу;
    plotting  -- plot_marker/plot_window: отрисовка ЭЭГ.

Единственный способ читать .e -- через NicoletEReader. Он скрывает то, что
данные приходится собирать из двух библиотек сразу, потому что каждая по
отдельности ошибается: см. документацию модуля epi.reader.

Командная строка: python -m epi --help
"""

from .catalog import Catalog, DB_NAME
from .reader import ChannelNotFound, NicoletEReader, Segment, learn_type_names
from .stats import render, summary
from .windows import CANONICAL, TARGET_RATE, WindowSet

__version__ = "1.0.0"

__all__ = [
    "Catalog",
    "NicoletEReader",
    "WindowSet",
    "Segment",
    "ChannelNotFound",
    "learn_type_names",
    "summary",
    "render",
    "plot_marker",
    "plot_window",
    "CANONICAL",
    "TARGET_RATE",
    "DB_NAME",
]


def __getattr__(name):
    """Функции отрисовки подгружаются лениво, чтобы import epi не тянул matplotlib."""
    if name in ("plot_marker", "plot_window"):
        from . import plotting
        return getattr(plotting, name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
