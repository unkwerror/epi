"""Нарезка каталога на обучающую выборку из окон одинаковой формы.

    from epi import Catalog, WindowSet

    windows = WindowSet(Catalog("catalog.sqlite"))
    windows.build()                       # разметить окна
    windows.split()                       # раздать пациентов по фолдам
    x, y = windows.sample("train", n=512)  # массивы прямо в память

Три свойства этих данных определяют устройство модуля:

* запись состоит из сегментов, разделённых настоящими разрывами во времени,
  поэтому окно обязано лежать внутри одного сегмента и не пересекать границу;
* усечённые выгрузки добиты сплошными нулями, поэтому окно нужно проверять на
  плоский сигнал, прежде чем брать его в выборку;
* монтажи у записей разные, поэтому окна урезаются до 19 электродов схемы
  10-20, которые есть в каждой записи, и приводятся к общей частоте.

Разбиение делается по пациентам. Окна одной записи перекрываются во времени и
тривиально предсказываются друг из друга, поэтому случайное разбиение самих
окон протащило бы тестовую выборку в обучение.
"""

from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from .reader import NicoletEReader

# 19 электродов схемы 10-20, общие для всех записей коллекции
CANONICAL = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz",
]
TARGET_RATE = 256.0
SEIZURE_TYPES = ("Seizure",)

SCHEMA = """
CREATE TABLE IF NOT EXISTS windows (
    id           INTEGER PRIMARY KEY,
    recording_id INTEGER,
    segment      INTEGER,
    start_sec    REAL,
    length_sec   REAL,
    label        INTEGER,
    overlap_sec  REAL
);
CREATE TABLE IF NOT EXISTS splits (
    patient_key TEXT PRIMARY KEY,
    fold        TEXT
);
CREATE INDEX IF NOT EXISTS windows_by_label ON windows(label);
CREATE INDEX IF NOT EXISTS windows_by_recording ON windows(recording_id);
"""


def overlap(window_start, window_end, spans):
    """Сколько секунд окна попадает внутрь размеченных интервалов."""
    return sum(
        max(0.0, min(window_end, stop) - max(window_start, start))
        for start, stop in spans
    )


class WindowSet:
    """Окна фиксированной длины, разбиение по пациентам и выборка данных."""

    def __init__(self, catalog, length_sec=10.0, stride_sec=5.0,
                 min_overlap_sec=5.0, rate=TARGET_RATE, channels=None):
        """catalog -- объект Catalog.

        length_sec       -- длина окна;
        stride_sec       -- шаг между началами соседних окон;
        min_overlap_sec  -- сколько секунд приступа должно попасть в окно,
                            чтобы оно считалось положительным;
        rate             -- общая частота, к которой всё приводится;
        channels         -- список каналов; по умолчанию схема 10-20.
        """
        self.catalog = catalog
        self.conn = catalog.conn
        self.conn.executescript(SCHEMA)
        self.length_sec = length_sec
        self.stride_sec = stride_sec
        self.min_overlap_sec = min_overlap_sec
        self.rate = rate
        self.channels = list(channels) if channels else list(CANONICAL)
        self._readers = {}

    def __repr__(self):
        row = self.conn.execute(
            "SELECT COUNT(*) n, COALESCE(SUM(label), 0) p FROM windows"
        ).fetchone()
        return "<WindowSet %.0f с: %d окон, %d с приступом>" % (
            self.length_sec, row["n"], row["p"]
        )

    # ------------------------------------------------------------------
    # разметка

    def _seizure_spans(self, recording_id, segment):
        """Интервалы приступов одного сегмента в секундах от его начала."""
        spans = []
        for row in self.conn.execute(
            """SELECT segment_time_sec, duration_sec FROM events
               WHERE recording_id = ? AND segment = ? AND type IN (%s)"""
            % ",".join("?" * len(SEIZURE_TYPES)),
            (recording_id, segment) + SEIZURE_TYPES,
        ):
            start = row["segment_time_sec"]
            spans.append((start, start + (row["duration_sec"] or 0.0)))
        return spans

    def build(self, verbose=True):
        """Перебрать сегменты, нарезать их на окна и разметить приступы.

        Возвращает словарь: total, positive, positive_share, skipped_segments.
        """
        self.conn.execute("DELETE FROM windows")
        rows = []
        for segment in self.conn.execute(
            """SELECT segments.*, recordings.id AS rec FROM segments
               JOIN recordings ON recordings.id = segments.recording_id
               WHERE recordings.deleted_at IS NULL"""
        ).fetchall():
            spans = self._seizure_spans(segment["rec"], segment["idx"])
            start = 0.0
            while start + self.length_sec <= segment["duration_sec"]:
                hit = overlap(start, start + self.length_sec, spans)
                rows.append((segment["rec"], segment["idx"], start, self.length_sec,
                             int(hit >= self.min_overlap_sec), hit))
                start += self.stride_sec

        self.conn.executemany(
            """INSERT INTO windows (recording_id, segment, start_sec, length_sec,
                                    label, overlap_sec)
               VALUES (?, ?, ?, ?, ?, ?)""", rows,
        )
        self.conn.commit()

        positive = sum(r[4] for r in rows)
        skipped = self.conn.execute(
            """SELECT COUNT(*) n FROM segments JOIN recordings
               ON recordings.id = segments.recording_id
               WHERE recordings.deleted_at IS NULL AND segments.duration_sec < ?""",
            (self.length_sec,),
        ).fetchone()["n"]
        report = {
            "total": len(rows),
            "positive": positive,
            "positive_share": 100 * positive / max(len(rows), 1),
            "skipped_segments": skipped,
        }
        if verbose:
            print("окна по %.0f с с шагом %.0f с: %d всего, %d с приступом (%.2f%%)"
                  % (self.length_sec, self.stride_sec, report["total"],
                     report["positive"], report["positive_share"]))
            print("сегментов короче одного окна: %d" % skipped)
        return report

    def split(self, seed=0, holdout=0.15, verbose=True):
        """Раздать пациентов целиком по фолдам, разнося носителей приступов.

        Положительные окна сосредоточены у нескольких пациентов, поэтому
        случайное групповое разбиение легко оставляет фолд вовсе без приступов.
        Пациенты раздаются от самых богатых приступами к бедным, и каждый
        уходит в тот фолд, который сильнее прочих отстаёт от своей доли.

        Возвращает список словарей по фолдам.
        """
        patients = self.conn.execute(
            """SELECT patient_key,
                      SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END) AS positives,
                      COUNT(*) AS windows
               FROM windows
               JOIN recordings ON recordings.id = windows.recording_id
               GROUP BY patient_key"""
        ).fetchall()

        generator = np.random.default_rng(seed)
        order = generator.permutation(len(patients))
        ranked = sorted(
            ((patients[i], int(order[i])) for i in range(len(patients))),
            key=lambda item: (-item[0]["positives"], item[1]),
        )

        targets = {"train": 1 - 2 * holdout, "val": holdout, "test": holdout}
        have = {fold: 0.0 for fold in targets}
        assignment = {}
        for row, _ in ranked:
            total = sum(have.values()) + max(row["positives"], 1)
            fold = min(targets, key=lambda f: have[f] - targets[f] * total)
            assignment[row["patient_key"]] = fold
            have[fold] += max(row["positives"], 1)

        self.conn.execute("DELETE FROM splits")
        self.conn.executemany("INSERT INTO splits VALUES (?, ?)", assignment.items())
        self.conn.commit()

        report = self.folds()
        if verbose:
            for row in report:
                print("%-6s %2d пациентов, %6d окон, %5d с приступом"
                      % (row["fold"], row["patients"], row["windows"],
                         row["positive"]))
        return report

    def folds(self):
        """Размер каждого фолда: пациенты, окна, положительные окна."""
        return self.catalog.query(
            """SELECT fold, COUNT(DISTINCT patient_key) AS patients,
                      COUNT(*) AS windows, COALESCE(SUM(label), 0) AS positive
               FROM windows
               JOIN recordings ON recordings.id = windows.recording_id
               JOIN splits USING (patient_key)
               GROUP BY fold ORDER BY windows DESC"""
        )

    def windows(self, fold=None, label=None, limit=None):
        """Строки окон с фильтрами по фолду и метке."""
        where, params = [], []
        if fold is not None:
            where.append("fold = ?")
            params.append(fold)
        if label is not None:
            where.append("label = ?")
            params.append(label)
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        sql = ("""SELECT windows.*, recordings.rel_path, recordings.patient_key
                  FROM windows
                  JOIN recordings ON recordings.id = windows.recording_id
                  LEFT JOIN splits USING (patient_key) %s""" % clause)
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return self.catalog.query(sql, params)

    # ------------------------------------------------------------------
    # чтение данных

    def _reader(self, recording_id):
        """Открытые ридеры кешируются: разбор заголовка стоит дороже чтения окна."""
        if recording_id not in self._readers:
            self._readers[recording_id] = NicoletEReader(
                self.catalog.path_of(recording_id), type_names={}
            )
        return self._readers[recording_id]

    def load(self, window):
        """Прочитать одно окно как массив (каналы, отсчёты) типа float32."""
        reader = self._reader(window["recording_id"])
        reader.select_segment(window["segment"])
        data = reader.read(self.channels, start_sec=window["start_sec"],
                           duration_sec=window["length_sec"])
        if reader.sampling_rate != self.rate:
            # дробь из двух частот, а не из их частного: частное -- двоичное
            # число с плавающей точкой, точное отношение которого имеет
            # астрономический знаменатель
            ratio = Fraction(int(self.rate), int(reader.sampling_rate))
            data = resample_poly(data, ratio.numerator, ratio.denominator, axis=0)
        expected = int(round(window["length_sec"] * self.rate))
        return data[:expected].T.astype(np.float32)

    @staticmethod
    def flat_fraction(data):
        """Доля отсчётов, где все электроды показывают ровно ноль."""
        return float(np.mean(np.all(data == 0, axis=0)))

    def sample(self, fold="train", n=1000, positive_ratio=0.5, seed=0,
               max_flat=0.01, verbose=True):
        """Собрать сбалансированную выборку окон в память.

        Классы набираются по отдельности, каждый до своей квоты, иначе редкие
        приступы утонули бы в случайной выборке. Окна с долей плоского сигнала
        выше max_flat отбрасываются.

        Возвращает (x, y): x формы (окна, каналы, отсчёты), y -- метки 0/1.
        """
        generator = np.random.default_rng(seed)

        def candidates(label):
            rows = self.conn.execute(
                """SELECT windows.* FROM windows
                   JOIN recordings ON recordings.id = windows.recording_id
                   JOIN splits USING (patient_key)
                   WHERE fold = ? AND label = ?""", (fold, label),
            ).fetchall()
            return [rows[i] for i in generator.permutation(len(rows))]

        def fill(label, quota):
            chosen, skipped = [], 0
            for window in candidates(label):
                if len(chosen) >= quota:
                    break
                data = self.load(window)
                if self.flat_fraction(data) > max_flat:
                    skipped += 1
                    continue
                chosen.append(data)
            return chosen, skipped

        wanted_positive = int(round(n * positive_ratio))
        positives, dropped_positive = fill(1, wanted_positive)
        negatives, dropped_negative = fill(0, n - wanted_positive)

        kept = positives + negatives
        if not kept:
            raise ValueError("после отсева плоского сигнала не осталось ни одного окна")
        labels = [1] * len(positives) + [0] * len(negatives)

        order = generator.permutation(len(kept))
        x = np.stack([kept[i] for i in order])
        y = np.array([labels[i] for i in order])

        if verbose:
            print("%s: %d окон (%d с приступом), отброшено плоских %d"
                  % (fold, len(y), int(y.sum()),
                     dropped_positive + dropped_negative))
            print("   форма %s при %g Гц" % (x.shape, self.rate))
        return x, y

    def export(self, fold="train", out_dir="training", **kwargs):
        """Сохранить выборку фолда в файл .npz.

        Внутри лежат x, y, channels и rate. Остальные аргументы такие же,
        как у sample().
        """
        x, y = self.sample(fold=fold, **kwargs)
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True)
        path = out_dir / ("%s.npz" % fold)
        np.savez_compressed(path, x=x, y=y,
                            channels=np.array(self.channels), rate=self.rate)
        print("записано в %s" % path)
        return path
