"""Выжимка статистики по каталогу.

    from epi import Catalog, summary, render
    report = summary(Catalog("catalog.sqlite"))
    render(report)

Всё считается по каталогу, поэтому стоит миллисекунды и не трогает файлы .e.
Числа охватывают только живые записи: помеченные удалёнными исключены, но
остаются в базе.
"""

from datetime import datetime

from .catalog import LIVE


def scalar(conn, query, *params):
    row = conn.execute(query, params).fetchone()
    return row[0] if row and row[0] is not None else 0


def quantiles(values, points=(0, 25, 50, 75, 100)):
    """Квантили списка; ключи вида p0, p25, ... p100."""
    if not values:
        return {}
    ordered = sorted(values)
    result = {}
    for point in points:
        index = min(len(ordered) - 1, int(round(point / 100 * (len(ordered) - 1))))
        result["p%d" % point] = round(ordered[index], 2)
    return result


def summary(catalog):
    """Собрать полный отчёт по каталогу в виде вложенного словаря.

    Разделы: totals, protocols, recordings_per_patient, segment_seconds,
    markers, seizures, unnamed_marker_classes, age_years_at_recording и, если
    окна уже размечены, windows.
    """
    conn = catalog.conn
    report = {}

    report["totals"] = {
        "patients": scalar(conn, "SELECT COUNT(DISTINCT patient_key) FROM recordings "
                                 "WHERE %s" % LIVE),
        "recordings": scalar(conn, "SELECT COUNT(*) FROM recordings WHERE %s" % LIVE),
        "segments": scalar(conn, "SELECT SUM(n_segments) FROM recordings WHERE %s"
                           % LIVE),
        "hours": round(scalar(conn, "SELECT SUM(duration_sec) FROM recordings "
                                    "WHERE %s" % LIVE) / 3600, 2),
        "deleted_recordings": scalar(
            conn, "SELECT COUNT(*) FROM recordings WHERE deleted_at IS NOT NULL"),
    }

    report["protocols"] = [
        {"sampling_rate": row["sampling_rate"], "n_channels": row["n_channels"],
         "recordings": row["n"], "hours": round(row["hours"] / 3600, 2)}
        for row in conn.execute(
            """SELECT sampling_rate, n_channels, COUNT(*) n,
                      SUM(duration_sec) hours
               FROM recordings WHERE %s
               GROUP BY sampling_rate, n_channels ORDER BY n DESC""" % LIVE)
    ]

    report["recordings_per_patient"] = quantiles([
        row[0] for row in conn.execute(
            "SELECT COUNT(*) FROM recordings WHERE %s GROUP BY patient_key" % LIVE)
    ])

    durations = [row[0] for row in conn.execute(
        """SELECT segments.duration_sec FROM segments
           JOIN recordings ON recordings.id = segments.recording_id
           WHERE %s""" % LIVE)]
    report["segment_seconds"] = quantiles(durations)
    report["segment_seconds"]["under_60s"] = sum(1 for d in durations if d < 60)
    report["segment_seconds"]["count"] = len(durations)

    report["markers"] = {
        row["type"]: row["n"] for row in conn.execute(
            """SELECT events.type, COUNT(*) n FROM events
               JOIN recordings ON recordings.id = events.recording_id
               WHERE %s GROUP BY events.type ORDER BY n DESC""" % LIVE)
    }

    seizures = [row[0] or 0.0 for row in conn.execute(
        """SELECT events.duration_sec FROM events
           JOIN recordings ON recordings.id = events.recording_id
           WHERE %s AND events.type = 'Seizure'""" % LIVE)]
    report["seizures"] = {
        "count": len(seizures),
        "total_minutes": round(sum(seizures) / 60, 1),
        "seconds": quantiles(seizures),
        "patients_with_any": scalar(
            conn, """SELECT COUNT(DISTINCT patient_key) FROM events
                     JOIN recordings ON recordings.id = events.recording_id
                     WHERE %s AND events.type = 'Seizure'""" % LIVE),
        "share_of_signal": round(
            100 * sum(seizures)
            / max(scalar(conn, "SELECT SUM(duration_sec) FROM recordings WHERE %s"
                         % LIVE), 1), 3),
    }

    report["unnamed_marker_classes"] = [
        {"guid": row["guid"], "count": row["n"],
         "authors": row["users"], "mean_seconds": round(row["d"] or 0, 1)}
        for row in conn.execute(
            """SELECT events.guid, COUNT(*) n, AVG(events.duration_sec) d,
                      GROUP_CONCAT(DISTINCT events.user) users
               FROM events JOIN recordings ON recordings.id = events.recording_id
               WHERE %s AND events.type = 'UNKNOWN'
               GROUP BY events.guid ORDER BY n DESC LIMIT 5""" % LIVE)
    ]

    ages = []
    for row in conn.execute(
        """SELECT patients.dob, MIN(segments.start_time) start
           FROM segments
           JOIN recordings ON recordings.id = segments.recording_id
           JOIN patients USING (patient_key)
           WHERE %s AND patients.dob IS NOT NULL
           GROUP BY patient_key""" % LIVE
    ):
        try:
            born = datetime.fromisoformat(row["dob"])
            began = datetime.fromisoformat(row["start"])
        except (TypeError, ValueError):
            continue
        ages.append((began - born).days / 365.25)
    report["age_years_at_recording"] = quantiles(ages)
    report["age_years_at_recording"]["negative_or_zero"] = sum(
        1 for age in ages if age <= 0)

    if conn.execute("SELECT name FROM sqlite_master WHERE name='windows'").fetchone():
        report["windows"] = [
            {"fold": row["fold"], "patients": row["k"], "windows": row["w"],
             "positive": row["p"] or 0, "excluded": row["g"] or 0}
            for row in conn.execute(
                """SELECT fold, SUM(label >= 0) w, SUM(label = 1) p,
                          SUM(label < 0) g, COUNT(DISTINCT patient_key) k
                   FROM windows
                   JOIN recordings ON recordings.id = windows.recording_id
                   JOIN splits USING (patient_key)
                   GROUP BY fold""")
        ]
    return report


def render(report):
    """Напечатать отчёт summary() в читаемом виде."""
    totals = report["totals"]
    print("пациентов %d | записей %d | сегментов %d | %.1f часов"
          % (totals["patients"], totals["recordings"], totals["segments"],
             totals["hours"]))
    if totals["deleted_recordings"]:
        print("  (%d записей помечены удалёнными и в числа выше не вошли)"
              % totals["deleted_recordings"])

    print("\nпротоколы съёмки:")
    for row in report["protocols"]:
        print("    %g Гц, %2d каналов   %2d записей, %6.1f ч"
              % (row["sampling_rate"], row["n_channels"], row["recordings"],
                 row["hours"]))

    seg = report["segment_seconds"]
    print("\nсегменты: %d, медиана %.1f с, самый длинный %.0f с, короче минуты %d"
          % (seg["count"], seg["p50"], seg["p100"], seg["under_60s"]))

    age = report["age_years_at_recording"]
    if age:
        print("возраст на момент записи: от %.1f до %.1f лет, медиана %.1f"
              % (age["p0"], age["p100"], age["p50"]))
        if age["negative_or_zero"]:
            print("    у %d пациентов дата рождения не раньше самой записи"
                  % age["negative_or_zero"])

    sz = report["seizures"]
    print("\nприступы: %d маркеров, всего %.1f минут, %.2f%% всего сигнала"
          % (sz["count"], sz["total_minutes"], sz["share_of_signal"]))
    print("    хотя бы один есть у %d из %d пациентов"
          % (sz["patients_with_any"], totals["patients"]))
    print("    длительность: медиана %.1f с, самый долгий %.0f с"
          % (sz["seconds"]["p50"], sz["seconds"]["p100"]))

    print("\nтипы маркеров:")
    for kind, count in list(report["markers"].items())[:10]:
        print("    %-38s %5d" % (kind, count))

    print("\nсамые крупные классы маркеров, оставшихся без названия:")
    for row in report["unnamed_marker_classes"]:
        print("    %-40s %4d  от %-18s в среднем %.1f с"
              % (row["guid"], row["count"], (row["authors"] or "")[:18],
                 row["mean_seconds"]))

    if report.get("windows"):
        print("\nобучающие окна:")
        for row in report["windows"]:
            print("    %-6s %2d пациентов, %6d окон, %5d положительных%s"
                  % (row["fold"], row["patients"], row["windows"], row["positive"],
                     ", %d в зазоре у приступа" % row["excluded"]
                     if row["excluded"] else ""))
