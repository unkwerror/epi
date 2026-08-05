"""Каталог записей .e поверх SQLite.

    from epi import Catalog

    catalog = Catalog("catalog.sqlite")
    catalog.sync("/путь/к/данным")     # обойти папки и обновить каталог
    catalog.patients()                 # список пациентов
    reader = catalog.open(recording_id)  # вернуться к сырому сигналу

Пациенты опознаются по GUID из заголовка файла, а не по имени папки: одна и та
же подпись папки (Patient1) на деле покрывала четырёх разных людей.

Записи, исчезнувшие с диска, помечаются удалёнными, но не стираются, чтобы
обучение можно было проследить до тех данных, на которых оно шло.

Все методы выборки возвращают списки словарей -- их можно сразу отдать в
pandas.DataFrame или обойти обычным циклом.
"""

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .reader import NicoletEReader, learn_type_names

DB_NAME = "catalog.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    patient_key TEXT PRIMARY KEY,
    guid        TEXT,
    alt_id      TEXT,
    dob         TEXT,
    sex_id      TEXT,
    notes       TEXT,
    first_seen  TEXT
);

CREATE TABLE IF NOT EXISTS recordings (
    id            INTEGER PRIMARY KEY,
    rel_path      TEXT UNIQUE,
    folder        TEXT,
    patient_key   TEXT REFERENCES patients(patient_key),
    file_size     INTEGER,
    file_mtime    REAL,
    content_hash  TEXT,
    sampling_rate REAL,
    n_channels    INTEGER,
    n_segments    INTEGER,
    duration_sec  REAL,
    ingested_at   TEXT,
    deleted_at    TEXT
);

CREATE TABLE IF NOT EXISTS segments (
    recording_id INTEGER REFERENCES recordings(id) ON DELETE CASCADE,
    idx          INTEGER,
    start_time   TEXT,
    offset_sec   REAL,
    duration_sec REAL,
    n_samples    INTEGER,
    PRIMARY KEY (recording_id, idx)
);

CREATE TABLE IF NOT EXISTS channels (
    recording_id INTEGER REFERENCES recordings(id) ON DELETE CASCADE,
    idx          INTEGER,
    name         TEXT,
    reference    TEXT,
    PRIMARY KEY (recording_id, idx)
);

CREATE TABLE IF NOT EXISTS events (
    id               INTEGER PRIMARY KEY,
    recording_id     INTEGER REFERENCES recordings(id) ON DELETE CASCADE,
    segment          INTEGER,
    time_sec         REAL,
    segment_time_sec REAL,
    duration_sec     REAL,
    type             TEXT,
    text             TEXT,
    guid             TEXT,
    user             TEXT,
    type_neo         TEXT,
    type_pynicolet   TEXT
);

CREATE TABLE IF NOT EXISTS event_types (
    guid       TEXT PRIMARY KEY,
    name       TEXT,
    learned_at TEXT
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE INDEX IF NOT EXISTS events_by_type ON events(type);
CREATE INDEX IF NOT EXISTS events_by_guid ON events(guid);
CREATE INDEX IF NOT EXISTS events_by_recording ON events(recording_id, segment);
CREATE INDEX IF NOT EXISTS recordings_by_patient ON recordings(patient_key);
"""

LIVE = "recordings.deleted_at IS NULL"


def quick_hash(path, edge=1 << 20):
    """Отпечаток по размеру файла и обоим его концам, чтобы не читать гигабайты."""
    size = path.stat().st_size
    digest = hashlib.blake2b(str(size).encode(), digest_size=16)
    with path.open("rb") as handle:
        digest.update(handle.read(edge))
        if size > edge:
            handle.seek(max(edge, size - edge))
            digest.update(handle.read(edge))
    return digest.hexdigest()


class Catalog:
    """Каталог записей: обход папок, поиск по метаданным, доступ к сигналу."""

    def __init__(self, db_path=DB_NAME, root=None):
        """db_path -- файл базы, root -- корень с папками пациентов.

        Корень запоминается при sync(), поэтому при повторном открытии
        каталога его можно не указывать.
        """
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        if root is not None:
            self._set_meta("root", str(Path(root).resolve()))

    def __repr__(self):
        totals = self.totals()
        return "<Catalog %s: %d пациентов, %d записей, %.1f ч>" % (
            self.db_path, totals["patients"], totals["recordings"], totals["hours"]
        )

    # ------------------------------------------------------------------
    # служебное

    def _set_meta(self, key, value):
        self.conn.execute(
            "INSERT INTO meta VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value)
        )
        self.conn.commit()

    def _get_meta(self, key, default=None):
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else default

    @property
    def root(self):
        """Корневая папка с данными, запомненная при последней синхронизации."""
        return Path(self._get_meta("root", "."))

    def query(self, sql, params=()):
        """Произвольный SQL к каталогу; возвращает список словарей."""
        return [dict(row) for row in self.conn.execute(sql, params)]

    # ------------------------------------------------------------------
    # наполнение

    def type_names(self):
        """Выученное соответствие GUID -> название типа события."""
        return {row["guid"]: row["name"]
                for row in self.conn.execute("SELECT guid, name FROM event_types")}

    def _upsert_patient(self, key, info, now):
        self.conn.execute(
            """INSERT INTO patients (patient_key, guid, alt_id, dob, sex_id, notes,
                                     first_seen)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(patient_key) DO UPDATE SET
                   alt_id = excluded.alt_id,
                   dob    = excluded.dob,
                   notes  = excluded.notes""",
            (key, info["guid"], info["alt_id"], info["dob"], info["sex_id"],
             info["notes"], now),
        )
        return key

    def ingest(self, path, rel_path, digest, now, type_names=None):
        """Разобрать один файл .e в каталог, заменив прошлую его версию."""
        reader = NicoletEReader(path, type_names=type_names)
        key = self._upsert_patient(reader.patient_key, reader.patient, now)
        stat = path.stat()

        self.conn.execute("DELETE FROM recordings WHERE rel_path = ?", (rel_path,))
        cursor = self.conn.execute(
            """INSERT INTO recordings (rel_path, folder, patient_key, file_size,
                                       file_mtime, content_hash, sampling_rate,
                                       n_channels, n_segments, duration_sec,
                                       ingested_at, deleted_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)""",
            (rel_path, path.parent.name, key, stat.st_size, stat.st_mtime, digest,
             reader.sampling_rate, len(reader.channels), len(reader.segments),
             reader.recorded_duration_sec, now),
        )
        recording_id = cursor.lastrowid

        self.conn.executemany(
            "INSERT INTO segments VALUES (?, ?, ?, ?, ?, ?)",
            [(recording_id, s.index, s.start_time.isoformat() if s.start_time else None,
              s.offset_sec, s.duration_sec, s.n_samples) for s in reader.segments],
        )
        self.conn.executemany(
            "INSERT INTO channels VALUES (?, ?, ?, ?)",
            [(recording_id, i, name, reference) for i, (name, reference)
             in enumerate(zip(reader.channels, reader.references))],
        )
        self.conn.executemany(
            """INSERT INTO events (recording_id, segment, time_sec, segment_time_sec,
                                   duration_sec, type, text, guid, user, type_neo,
                                   type_pynicolet)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [(recording_id, e["segment"], e["time_sec"], e["segment_time_sec"],
              e["duration_sec"], e["type"], e["text"], e["guid"], e["user"],
              e["type_neo"], e["type_pynicolet"]) for e in reader.read_events()],
        )
        return recording_id

    def sync(self, root=None, verbose=True):
        """Обойти папку и привести каталог в соответствие с диском.

        Файл разбирается заново, только если изменились его размер или время
        правки, а отпечаток содержимого при этом тоже разошёлся. Записи,
        пропавшие с диска, помечаются удалёнными.

        Возвращает словарь со счётчиками: added, updated, unchanged, revived,
        deleted.
        """
        root = Path(root).resolve() if root is not None else self.root
        self._set_meta("root", str(root))
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")

        known = {row["rel_path"]: row
                 for row in self.conn.execute("SELECT * FROM recordings")}
        seen = set()
        counts = {"added": 0, "updated": 0, "unchanged": 0, "revived": 0, "deleted": 0}

        files = sorted(root.rglob("*.e"))
        type_names = learn_type_names(files)
        self.conn.executemany(
            """INSERT INTO event_types VALUES (?, ?, ?)
               ON CONFLICT(guid) DO UPDATE SET name = excluded.name,
                                               learned_at = excluded.learned_at""",
            [(guid, name, now) for guid, name in type_names.items()],
        )

        for path in files:
            rel_path = str(path.relative_to(root))
            seen.add(rel_path)
            stat = path.stat()
            row = known.get(rel_path)

            if row and row["deleted_at"] is None \
                    and row["file_size"] == stat.st_size \
                    and abs(row["file_mtime"] - stat.st_mtime) < 1e-6:
                counts["unchanged"] += 1
                continue

            digest = quick_hash(path)
            if row and row["content_hash"] == digest:
                # сдвинулось только время правки, либо файл вернулся после удаления
                self.conn.execute(
                    "UPDATE recordings SET file_mtime = ?, deleted_at = NULL "
                    "WHERE id = ?", (stat.st_mtime, row["id"]),
                )
                counts["revived"] += row["deleted_at"] is not None
                counts["unchanged"] += row["deleted_at"] is None
                continue

            self.ingest(path, rel_path, digest, now, type_names)
            counts["added"] += row is None
            counts["updated"] += row is not None

        gone = [p for p, row in known.items()
                if p not in seen and row["deleted_at"] is None]
        self.conn.executemany(
            "UPDATE recordings SET deleted_at = ? WHERE rel_path = ?",
            [(now, p) for p in gone],
        )
        counts["deleted"] = len(gone)
        self.conn.commit()

        if verbose:
            print("синхронизация %s -> %s" % (root, self.db_path))
            print("  добавлено %(added)d, обновлено %(updated)d, без изменений "
                  "%(unchanged)d, возвращено %(revived)d, помечено удалёнными "
                  "%(deleted)d" % counts)
        return counts

    # ------------------------------------------------------------------
    # выборки

    def totals(self):
        """Общие числа по каталогу: пациенты, записи, сегменты, часы."""
        row = self.conn.execute(
            """SELECT COUNT(*) AS recordings,
                      COUNT(DISTINCT patient_key) AS patients,
                      COALESCE(SUM(n_segments), 0) AS segments,
                      COALESCE(SUM(duration_sec), 0) / 3600 AS hours
               FROM recordings WHERE %s""" % LIVE
        ).fetchone()
        return dict(row)

    def patients(self):
        """Все пациенты со сводкой по их записям."""
        return self.query(
            """SELECT patients.patient_key, patients.alt_id, patients.dob,
                      patients.sex_id, patients.notes,
                      COUNT(recordings.id) AS recordings,
                      COALESCE(SUM(recordings.duration_sec), 0) / 3600 AS hours,
                      (SELECT COUNT(*) FROM events
                       JOIN recordings r2 ON r2.id = events.recording_id
                       WHERE r2.patient_key = patients.patient_key
                         AND events.type = 'Seizure') AS seizures
               FROM patients
               LEFT JOIN recordings ON recordings.patient_key = patients.patient_key
                    AND recordings.deleted_at IS NULL
               GROUP BY patients.patient_key
               ORDER BY seizures DESC, hours DESC"""
        )

    def patient(self, key):
        """Всё, что известно об одном пациенте.

        key можно указать началом ключа -- достаточно первых символов GUID.
        Возвращает словарь с полями пациента и списками recordings и seizures.
        """
        row = self.conn.execute(
            "SELECT * FROM patients WHERE patient_key LIKE ?", (key + "%",)
        ).fetchone()
        if row is None:
            raise KeyError("нет пациента, начинающегося с %r" % key)

        info = dict(row)
        info["recordings"] = self.recordings(patient_key=row["patient_key"])
        info["seizures"] = self.query(
            """SELECT events.*, recordings.folder FROM events
               JOIN recordings ON recordings.id = events.recording_id
               WHERE recordings.patient_key = ? AND %s AND events.type = 'Seizure'
               ORDER BY events.recording_id, events.time_sec""" % LIVE,
            (row["patient_key"],),
        )
        return info

    def recordings(self, patient_key=None, include_deleted=False):
        """Записи каталога, при желании только по одному пациенту."""
        where = [] if include_deleted else [LIVE]
        params = []
        if patient_key is not None:
            where.append("recordings.patient_key = ?")
            params.append(patient_key)
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        return self.query(
            "SELECT * FROM recordings %s ORDER BY rel_path" % clause, params
        )

    def segments(self, recording_id):
        """Сегменты одной записи."""
        return self.query(
            "SELECT * FROM segments WHERE recording_id = ? ORDER BY idx",
            (recording_id,),
        )

    def channels(self, recording_id):
        """Каналы одной записи."""
        return self.query(
            "SELECT * FROM channels WHERE recording_id = ? ORDER BY idx",
            (recording_id,),
        )

    def events(self, type=None, guid=None, recording_id=None, patient_key=None,
               limit=None, offset=0):
        """Маркеры с фильтрами по типу, GUID, записи или пациенту.

        К каждому маркеру подмешиваются rel_path и folder его записи, поэтому
        результат можно сразу передать в open_event() или в plot_marker().
        """
        where, params = [LIVE], []
        for column, value in (("events.type", type), ("events.guid", guid),
                              ("events.recording_id", recording_id),
                              ("recordings.patient_key", patient_key)):
            if value is not None:
                where.append("%s = ?" % column)
                params.append(value)
        sql = ("""SELECT events.*, recordings.rel_path, recordings.folder,
                         recordings.patient_key
                  FROM events JOIN recordings ON recordings.id = events.recording_id
                  WHERE %s ORDER BY events.recording_id, events.time_sec"""
               % " AND ".join(where))
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params += [limit, offset]
        return self.query(sql, params)

    def event_counts(self):
        """Сколько маркеров каждого типа и сколько всего времени они занимают."""
        return self.query(
            """SELECT events.type, COUNT(*) AS n,
                      COALESCE(SUM(events.duration_sec), 0) AS seconds
               FROM events JOIN recordings ON recordings.id = events.recording_id
               WHERE %s GROUP BY events.type ORDER BY n DESC""" % LIVE
        )

    # ------------------------------------------------------------------
    # мост к сигналу

    def path_of(self, recording):
        """Полный путь к файлу .e по номеру записи или по её строке."""
        if isinstance(recording, dict):
            rel_path = recording["rel_path"]
        else:
            row = self.conn.execute(
                "SELECT rel_path FROM recordings WHERE id = ?", (recording,)
            ).fetchone()
            if row is None:
                raise KeyError("нет записи с id=%r" % recording)
            rel_path = row["rel_path"]
        return self.root / rel_path

    def open(self, recording, segment=0):
        """Открыть запись ридером, подставив выученные названия типов событий.

        recording -- номер записи или любой словарь с полем rel_path
        (например строка из events()).
        """
        return NicoletEReader(
            self.path_of(recording), segment=segment, type_names=self.type_names()
        )

    def open_event(self, event, pad_sec=5.0, window_sec=25.0):
        """Прочитать сигнал вокруг маркера.

        Возвращает (reader, data, times): data -- массив (отсчёты, каналы) со
        всеми каналами, times -- ось времени в секундах от начала сегмента.
        Длинные маркеры обрезаются до window_sec от их начала.
        """
        reader = self.open(event, segment=event["segment"])
        start = max(0.0, event["segment_time_sec"] - pad_sec)
        span = min((event["duration_sec"] or 0) + 2 * pad_sec, window_sec,
                   reader.duration_sec - start)
        data = reader.read(start_sec=start, duration_sec=span)
        return reader, data, reader.times(start_sec=start, duration_sec=span)

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
