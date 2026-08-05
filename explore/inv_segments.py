"""Inventory step 2: the time structure of every recording.

A .e file holds one or more contiguous segments, each with its own wall-clock
start. Gaps between them are real breaks in the recording, so segments are the
atomic unit for anything that slices signal.

The overlap check at the end matters for the crawler: the same patient may be
exported more than once, and two exports covering the same wall-clock interval
would otherwise enter the dataset twice.
"""

import sys
from collections import defaultdict
from pathlib import Path

from inv_patients import patient_info, patient_key
from epi import NicoletEReader


def describe(path):
    reader = NicoletEReader(path)
    return {
        "path": path,
        "reader": reader,
        "patient": patient_key(patient_info(path)),
        "segments": reader.segments,
    }


def main(root):
    records = [describe(p) for p in sorted(Path(root).rglob("*.e"))]

    total_hours = 0.0
    total_segments = 0
    print("%-46s %4s %8s %9s  %s" % ("recording", "segs", "hours", "gap_h", "starts"))
    for rec in records:
        segs = rec["segments"]
        hours = sum(s.duration_sec for s in segs) / 3600
        span = (segs[-1].offset_sec + segs[-1].duration_sec) / 3600
        total_hours += hours
        total_segments += len(segs)
        print("%-46s %4d %8.2f %9.2f  %s -> %s"
              % (rec["path"].parent.name, len(segs), hours, span - hours,
                 segs[0].start_time, segs[-1].start_time))

    print("\ntotal: %d recordings, %d segments, %.1f hours of signal"
          % (len(records), total_segments, total_hours))

    print("\nsegment length distribution (minutes):")
    lengths = sorted(s.duration_sec / 60 for r in records for s in r["segments"])
    for label, value in [
        ("min", lengths[0]),
        ("median", lengths[len(lengths) // 2]),
        ("max", lengths[-1]),
        ("under 1 min", sum(1 for v in lengths if v < 1)),
        ("under 5 min", sum(1 for v in lengths if v < 5)),
    ]:
        print("    %-12s %.2f" % (label, value))

    print("\noverlap check between exports of the same patient:")
    by_patient = defaultdict(list)
    for rec in records:
        by_patient[rec["patient"]].append(rec)
    clashes = 0
    for patient, group in by_patient.items():
        if len(group) < 2:
            continue
        intervals = [
            (s.start_time, s.start_time.timestamp() + s.duration_sec, rec, s)
            for rec in group
            for s in rec["segments"]
        ]
        intervals.sort(key=lambda item: item[0])
        for (start_a, end_a, rec_a, seg_a), (start_b, _, rec_b, seg_b) in zip(
            intervals, intervals[1:]
        ):
            if start_b.timestamp() < end_a:
                clashes += 1
                print("    %s: %s seg%d overlaps %s seg%d"
                      % (patient, rec_a["path"].parent.name, seg_a.index,
                         rec_b["path"].parent.name, seg_b.index))
    if not clashes:
        print("    none")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
