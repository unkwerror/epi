"""Чтение файлов Nicolet/Nervus (.e).

    from epi import NicoletEReader

    reader = NicoletEReader("Pruned.e")
    ekg = reader.read_channel("EKG", start_sec=0, duration_sec=60)
    events = reader.read_events()

Чтение оконное: с диска поднимается только запрошенный интервал, поэтому
минута из десятичасовой записи стоит несколько мегабайт памяти.

Файл .e может содержать несколько сегментов записи, разделённых разрывами
по реальному времени. Отсчёты читаются только внутри одного сегмента,
поэтому start_sec всегда отсчитывается от начала текущего сегмента
(см. .segments и .select_segment()), а время событий -- от начала всей
записи.

События собираются из двух библиотек, потому что каждая ошибается по-своему.
neo верно раскладывает маркеры по сегментам, но многие типы оставляет как
UNKNOWN и портит кириллицу; pynicolet называет больше типов и правильно
декодирует текст, но в многосегментных файлах прижимает маркеры к концу
записи. Списки совпадают один в один после отбрасывания служебных маркеров
Boundary -- это проверено на всех записях: начала приступов сходятся в тех же
позициях, а в односегментных файлах, где обеим шкалам времени можно верить,
расхождение не превышает 3 мс.
"""

from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from neo.rawio import NicoletRawIO
from pynicolet import NicoletReader


class ChannelNotFound(KeyError):
    """Запрошенного канала нет в записи (или он идёт с другой частотой)."""


# значения, которые означают "тип события не определён"
UNRESOLVED = ("UNKNOWN", "")


def _repair_text(text):
    """Обратить чтение строки UTF-16LE по одному байту на символ.

    Возвращает текст и признак того, что строку пришлось чинить. Починка
    неполная: neo вдобавок выбрасывает нулевой байт у каждого ASCII-символа,
    поэтому испорченную подпись лучше брать из pynicolet, когда она там есть.
    """
    if not text or ("\x04" not in text and "\x00" not in text):
        return text, False
    try:
        return text.encode("latin-1").decode("utf-16-le").rstrip("\x00").strip(), True
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text, True


def _to_datetime(date_parts, time_parts):
    try:
        year, month, day = (int(v) for v in date_parts)
        hour, minute, second = (int(v) for v in time_parts)
        return datetime(year, month, day, hour, minute, second)
    except (TypeError, ValueError):
        return None


class Segment:
    """Один непрерывный кусок записи.

    Атрибуты:
        index         -- номер сегмента в файле;
        duration_sec  -- длительность в секундах;
        n_samples     -- число отсчётов;
        start_time    -- астрономическое время начала (datetime или None);
        offset_sec    -- секунды от начала записи до начала сегмента.
    """

    def __init__(self, index, raw, offset_sec):
        self.index = index
        self.duration_sec = float(raw["duration"])
        self.n_samples = int(raw["sampleCount"])
        self.start_time = _to_datetime(raw["startDate"], raw["startTime"])
        # секунды от начала записи до начала этого сегмента, вместе с разрывами;
        # время событий отсчитывается от той же точки
        self.offset_sec = offset_sec

    def __repr__(self):
        return "<Segment %d: %.1f s from %s>" % (
            self.index,
            self.duration_sec,
            self.start_time,
        )


def learn_type_names(paths, rounds=4):
    """Построить словарь GUID -> название типа события по всей коллекции.

    Обучение по одному файлу слишком робкое: GUID остаётся безымянным в одной
    записи только потому, что там маркеры случайно совпали по времени, хотя в
    другой записи он определяется однозначно. Голоса собираются со всех файлов,
    побеждает самое частое название, а результат подаётся обратно на вход,
    чтобы имена, найденные в одном файле, помогали разобрать плотные группы
    маркеров в следующем.

    Возвращает словарь, который стоит передать во все ридеры через type_names.
    """
    readers = [NicoletEReader(path, type_names={}) for path in paths]
    known = {}
    for _ in range(rounds):
        votes = defaultdict(Counter)
        for reader in readers:
            for guid, name in reader._names_by_guid(known).items():
                votes[guid][name] += 1
        found = {guid: counter.most_common(1)[0][0] for guid, counter in votes.items()}
        if found == known:
            break
        known = found
    return known


class NicoletEReader:
    """Одна запись .e: сигнал, каналы, сегменты, события и данные пациента.

    Основные атрибуты:
        channels               -- имена каналов основного потока;
        references             -- референс каждого канала;
        excluded               -- каналы с другой частотой, [(имя, частота)];
        sampling_rate          -- частота дискретизации, Гц;
        segments               -- список Segment;
        recorded_duration_sec  -- сумма длительностей сегментов;
        total_duration_sec     -- от начала первого до конца последнего,
                                  вместе с разрывами;
        patient                -- словарь с данными пациента;
        patient_key            -- устойчивый идентификатор пациента.
    """

    def __init__(self, filename, segment=0, type_names=None):
        """Открыть файл .e.

        segment     -- номер сегмента, активного сразу после открытия;
        type_names  -- общий словарь GUID -> название из learn_type_names().
                       Он пропускает обучение на этом файле, а заодно избавляет
                       от открытия файла через neo. Пустой словарь {} отключает
                       определение типов совсем: это самый быстрый режим, но
                       часть маркеров останется UNKNOWN. Значение по умолчанию
                       None означает обучение по одному этому файлу.
        """
        self.path = Path(filename)
        self._type_names = type_names
        self._reader = NicoletReader(str(self.path))
        self._header = self._reader.read_header()
        self._events_backend = None
        self._merged = None

        raw_segments = self._header["Segments"]
        origin = _to_datetime(raw_segments[0]["startDate"], raw_segments[0]["startTime"])
        self.segments = []
        for i, raw in enumerate(raw_segments):
            begin = _to_datetime(raw["startDate"], raw["startTime"])
            offset = (begin - origin).total_seconds() if begin and origin else 0.0
            self.segments.append(Segment(i, raw, offset))
        last = self.segments[-1]
        self.total_duration_sec = last.offset_sec + last.duration_sec
        self.recorded_duration_sec = sum(s.duration_sec for s in self.segments)

        first = raw_segments[0]
        # pynicolet оставляет только каналы с преобладающей частотой;
        # столбец i из read_data() -- это канал matching[i] записи
        matching = list(self._header["matchingChannels"])
        self.channels = [first["chName"][i].strip() for i in matching]
        self.references = [first["refName"][i].strip() for i in matching]
        self.excluded = [
            (first["chName"][i].strip(), first["samplingRate"][i])
            for i in self._header["excludedChannels"]
        ]
        self.sampling_rate = float(self._header["targetSamplingRate"])
        self.patient = self._patient_info()
        # имена папок содержат номера выгрузок, а не личности, поэтому GUID из
        # заголовка -- единственный устойчивый ключ; altID подменяет его, когда
        # GUID отсутствует
        self.patient_key = self.patient["guid"] or "altid:%s" % self.patient["alt_id"]

        self.select_segment(segment)

    def __repr__(self):
        return "<NicoletEReader %s: %d ch, %g Hz, %d segment(s), %.1f min recorded>" % (
            self.path.parent.name,
            len(self.channels),
            self.sampling_rate,
            len(self.segments),
            self.recorded_duration_sec / 60,
        )

    def _patient_info(self):
        info = self._header["info"]
        dob = info.get("DOB") or []
        return {
            "guid": info.get("patientID"),
            "alt_id": info.get("altID"),
            "last_name": info.get("lastName"),
            "sex_id": info.get("sexID"),
            # в заголовке дата рождения лежит как [день, месяц, год]
            "dob": "%04d-%02d-%02d" % (dob[2], dob[1], dob[0])
            if len(dob) == 3 else None,
            "notes": (info.get("notes") or "").replace("\r\n", " | ").strip(),
        }

    def select_segment(self, index):
        """Переключить последующие чтения на другой сегмент."""
        if not 0 <= index < len(self.segments):
            raise IndexError(
                "segment %d out of range (file has %d)" % (index, len(self.segments))
            )
        self.segment = index
        current = self.segments[index]
        self.n_samples = current.n_samples
        self.duration_sec = current.duration_sec
        self.start_time = current.start_time
        return current

    def _index(self, name):
        wanted = name.strip().lower()
        for i, ch in enumerate(self.channels):
            if ch.lower() == wanted:
                return i
        for excluded_name, rate in self.excluded:
            if excluded_name.lower() == wanted:
                raise ChannelNotFound(
                    "%r is recorded at %g Hz instead of %g Hz and is not part of "
                    "the main signal stream" % (name, rate, self.sampling_rate)
                )
        raise ChannelNotFound(
            "%r not found; available channels: %s" % (name, ", ".join(self.channels))
        )

    def _sample_range(self, start_sec, duration_sec):
        start = int(round(start_sec * self.sampling_rate))
        if start < 0 or start >= self.n_samples:
            raise ValueError(
                "start_sec=%g is outside segment %d (0..%.1f s)"
                % (start_sec, self.segment, self.duration_sec)
            )
        if duration_sec is None:
            stop = self.n_samples
        else:
            stop = min(
                start + int(round(duration_sec * self.sampling_rate)), self.n_samples
            )
        # pynicolet ждёт диапазон [первый, последний] с нумерацией от единицы
        return [start + 1, stop]

    def read_channel(self, name, start_sec=0.0, duration_sec=None):
        """Вернуть один канал текущего сегмента в микровольтах.

        duration_sec=None читает сегмент до конца.
        """
        data = self._reader.read_data(
            segment=self.segment,
            chIdx=[self._index(name)],
            range_=self._sample_range(start_sec, duration_sec),
        )
        return data[:, 0]

    def read(self, names=None, start_sec=0.0, duration_sec=None):
        """Вернуть несколько каналов массивом (отсчёты, каналы) в микровольтах.

        names=None читает все каналы основного потока в их исходном порядке.
        """
        idx = None if names is None else [self._index(n) for n in names]
        return self._reader.read_data(
            segment=self.segment,
            chIdx=idx,
            range_=self._sample_range(start_sec, duration_sec),
        )

    def times(self, start_sec=0.0, duration_sec=None):
        """Ось времени в секундах для окна read() / read_channel()."""
        first, last = self._sample_range(start_sec, duration_sec)
        return np.arange(first - 1, last) / self.sampling_rate

    def _event_reader(self):
        if self._events_backend is None:
            backend = NicoletRawIO(filename=str(self.path))
            backend.parse_header()
            self._events_backend = backend
        return self._events_backend

    def _neo_names(self):
        """Времена маркеров и названия типов так, как их видит neo."""
        backend = self._event_reader()
        n_channels = len(backend.header["event_channels"])
        found = []
        for segment in range(len(self.segments)):
            for channel in range(n_channels):
                if not backend.event_count(0, segment, channel):
                    continue
                stamps, spans, labels = backend.get_event_timestamps(
                    0, segment, channel
                )
                times = backend.rescale_event_timestamp(
                    stamps, event_channel_index=channel
                )
                for i, label in enumerate(labels):
                    text, garbled = _repair_text(str(label))
                    # подпись, пришедшая как UTF-16, -- это свободный текст
                    # врача, а не название типа события
                    found.append((float(times[i]), "" if garbled else text))
        return found

    def _clusters(self):
        """Сгруппировать маркеры обеих библиотек, попавшие в одну секунду."""
        buckets = defaultdict(lambda: ([], []))
        for event in self._header["raw_events"]:
            buckets[round(self._offset_of(event))][0].append(event["GUID"])
        for time, name in self._neo_names():
            buckets[round(time)][1].append(name)
        return list(buckets.values())

    def _names_by_guid(self, known=None):
        """Вывести соответствие GUID -> название типа из данных neo.

        neo называет несколько типов, которые pynicolet отдаёт как UNKNOWN, но
        списки нельзя просто сложить попарно: маркеры с одинаковым временем
        выходят в произвольном порядке. Внутри группы одновременных маркеров имя
        берётся, только если оно вынужденное -- либо вся группа единодушна, либо
        все остальные маркеры в ней уже опознаны и на один GUID остаётся ровно
        одно имя. Всё, что осталось неоднозначным, оставляем неопределённым,
        а не угадываем.
        """
        known = known or {}
        deduced = {}
        for guids, names in self._clusters():
            if len(guids) != len(names):
                continue
            unique = set(names)
            if len(unique) == 1 and unique != {""}:
                for guid in guids:
                    deduced.setdefault(guid, names[0])
                continue

            remaining_names = list(names)
            remaining_guids = []
            for guid in guids:
                expected = known.get(guid) or deduced.get(guid)
                if expected in remaining_names:
                    remaining_names.remove(expected)
                else:
                    remaining_guids.append(guid)
            if len(remaining_guids) == 1 and len(remaining_names) == 1:
                deduced.setdefault(remaining_guids[0], remaining_names[0])
        return {g: n for g, n in deduced.items() if n not in UNRESOLVED}

    def _offset_of(self, event):
        """Секунды от начала записи до сырого маркера."""
        origin = self.segments[0].start_time
        if origin is None or event.get("date") is None:
            return 0.0
        return (event["date"] - origin).total_seconds()

    def _locate(self, offset):
        """В каком сегменте лежит маркер и на какой секунде внутри него."""
        for segment in reversed(self.segments):
            if offset >= segment.offset_sec:
                return segment.index, offset - segment.offset_sec
        return 0, offset

    def _merge_events(self):
        """Позиции берём из сырых абсолютных времён, названия -- из обеих библиотек."""
        by_guid = (
            self._type_names if self._type_names is not None else self._names_by_guid()
        )
        merged = []
        for event in self._header["raw_events"]:
            offset = self._offset_of(event)
            segment, local = self._locate(offset)
            from_pynicolet = event["IDStr"]
            from_neo = by_guid.get(event["GUID"], "")
            if from_pynicolet not in UNRESOLVED:
                kind = from_pynicolet
            elif from_neo not in UNRESOLVED:
                kind = from_neo
            else:
                kind = "UNKNOWN"
            merged.append(
                {
                    "type": kind,
                    "text": (event["annotation"] or event["label"] or "").strip(" -"),
                    "guid": event["GUID"],
                    "user": event["user"],
                    "time_sec": offset,
                    "segment": segment,
                    "segment_time_sec": local,
                    "duration_sec": event["duration"],
                    "type_neo": from_neo,
                    "type_pynicolet": from_pynicolet,
                }
            )
        merged.sort(key=lambda e: e["time_sec"])
        return merged

    def read_events(self, kind=None):
        """Маркеры, расставленные врачом, со временем в секундах.

        kind (например "Seizure") фильтрует по типу без учёта регистра.

        В каждом словаре:
            type              -- название типа события;
            text              -- подпись врача или отведение;
            guid              -- идентификатор типа события;
            user              -- кто поставил маркер;
            time_sec          -- секунды от начала записи, вместе с разрывами;
            segment           -- номер сегмента;
            segment_time_sec  -- секунды от начала этого сегмента;
            duration_sec      -- длительность маркера;
            type_neo,
            type_pynicolet    -- что сказала каждая библиотека по отдельности.

        Пара segment и segment_time_sec -- это готовые аргументы для
        select_segment() и read_channel().
        """
        if self._merged is None:
            self._merged = self._merge_events()
        if kind is None:
            return list(self._merged)
        wanted = kind.strip().lower()
        return [e for e in self._merged if e["type"].strip().lower() == wanted]
