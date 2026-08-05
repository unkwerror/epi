"""Отрисовка ЭЭГ вокруг маркера.

    from epi import Catalog, plot_marker
    catalog = Catalog("catalog.sqlite")
    event = catalog.events(type="Seizure", limit=1)[0]
    plot_marker(catalog, event)

Функции возвращают фигуру matplotlib, поэтому в ноутбуке график появляется
прямо в ячейке, а в скрипте его можно сохранить через figure.savefig().
"""

import matplotlib.pyplot as plt
import numpy as np

# порядок отведений как на классической раскладке, а не по алфавиту:
# соседние строки должны быть соседними электродами
TEN_TWENTY = [
    "Fp1", "F7", "T3", "T5", "O1", "Fp2", "F8", "T4", "T6", "O2",
    "F3", "C3", "P3", "F4", "C4", "P4", "Fz", "Cz", "Pz",
]


def plot_marker(catalog, event, pad_sec=5.0, window_sec=25.0, figsize=(14, 9)):
    """Нарисовать сигнал вокруг маркера классической этажеркой отведений.

    pad_sec     -- сколько секунд показать до начала и после конца маркера;
    window_sec  -- предел длины окна. Длинные маркеры рисуются только от начала:
                   минуты ЭЭГ, втиснутые в одну картинку, сливаются в кашу.

    Красная линия отмечает начало маркера, полоса -- его длительность.
    """
    reader = catalog.open(event, segment=event["segment"])
    start = max(0.0, event["segment_time_sec"] - pad_sec)
    span = min((event["duration_sec"] or 0) + 2 * pad_sec, window_sec,
               reader.duration_sec - start)

    names = [c for c in TEN_TWENTY if c in reader.channels]
    if "EKG" in reader.channels:
        names.append("EKG")
    data = reader.read(names, start_sec=start, duration_sec=span)
    times = np.arange(len(data)) / reader.sampling_rate + start

    # шаг между строками подбирается по самим данным, иначе кривые либо
    # налезают друг на друга, либо расплющиваются об фиксированный потолок
    spacing = max(20.0, float(np.percentile(np.abs(data - np.median(data)), 99)))
    figure, axes = plt.subplots(figsize=figsize)
    for i, name in enumerate(names):
        trace = data[:, i] - np.median(data[:, i])
        axes.plot(times, np.clip(trace, -spacing, spacing) - i * spacing,
                  linewidth=0.4, color="black")
    axes.set_yticks([-i * spacing for i in range(len(names))])
    axes.set_yticklabels(names, fontsize=8)
    axes.set_xlabel("секунды внутри сегмента %d" % event["segment"])
    axes.set_xlim(times[0], times[-1])

    onset = event["segment_time_sec"]
    axes.axvline(onset, color="crimson", linewidth=1.2)
    if event["duration_sec"]:
        axes.axvspan(onset, onset + event["duration_sec"],
                     color="crimson", alpha=0.10)

    axes.set_title(
        "%s | %s | %s\nпоставил %s, длительность %.1f с, на %.1f с от начала сегмента"
        % (event.get("folder", ""), event["type"],
           event["text"] or event["guid"], event["user"] or "-",
           event["duration_sec"] or 0, onset), fontsize=9)
    axes.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure


def plot_window(data, channels, rate, label=None, figsize=(12, 7)):
    """Нарисовать одно окно из выборки: массив (каналы, отсчёты).

    Подходит для того, чтобы глазами проверить, что уехало в обучение.
    """
    times = np.arange(data.shape[1]) / rate
    spacing = max(20.0, float(np.percentile(np.abs(data - np.median(data)), 99)))
    figure, axes = plt.subplots(figsize=figsize)
    for i, name in enumerate(channels):
        trace = data[i] - np.median(data[i])
        axes.plot(times, np.clip(trace, -spacing, spacing) - i * spacing,
                  linewidth=0.5, color="black")
    axes.set_yticks([-i * spacing for i in range(len(channels))])
    axes.set_yticklabels(channels, fontsize=8)
    axes.set_xlabel("секунды")
    axes.set_xlim(times[0], times[-1])
    if label is not None:
        axes.set_title("метка %d (%s)"
                       % (label, "приступ" if label else "фон"), fontsize=10)
    axes.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure
