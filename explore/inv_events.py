"""Inventory step 4: what the technicians actually marked.

Events come from neo through nicolet_e, since pynicolet cannot place markers
correctly in multi-segment files.

Besides counting marker types, this checks whether the very short segments of
the pruned exports are clipped around seizure markers, which would make them
ready-made positive examples rather than raw signal.
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

from epi import NicoletEReader

# markers describing the file itself rather than the patient
BOOKKEEPING = {"Exam start", "Review progress", "Prune", "Boundary", "Recording Off",
               "Recording On", "Impedance", "Montage"}


def main(root):
    per_file = {}
    for path in sorted(Path(root).rglob("*.e")):
        reader = NicoletEReader(path)
        per_file[path.parent.name] = (reader, reader.read_events())

    print("marker types across the whole collection:")
    kinds = Counter(e["type"] for _, events in per_file.values() for e in events)
    for kind, count in kinds.most_common():
        tag = "  (bookkeeping)" if kind in BOOKKEEPING else ""
        print("    %-22s %4d%s" % (kind, count, tag))

    print("\nclinical markers per recording:")
    print("%-46s %6s %8s %9s" % ("recording", "seizure", "sz_total_s", "patient_ev"))
    seizure_total = 0
    for name, (reader, events) in per_file.items():
        seizures = [e for e in events if e["type"] == "Seizure"]
        patient_events = [e for e in events if e["type"] == "Patient Event"]
        spans = sum(e["duration_sec"] or 0 for e in seizures)
        seizure_total += len(seizures)
        print("%-46s %6d %8.1f %9d" % (name, len(seizures), spans, len(patient_events)))
    print("total seizure markers: %d" % seizure_total)

    print("\nare short segments clipped around seizures?")
    print("%-46s %5s %7s %8s" % ("recording", "segs", "short", "with_sz"))
    for name, (reader, events) in per_file.items():
        short = [s for s in reader.segments if s.duration_sec < 60]
        if not short:
            continue
        seizures = [e for e in events if e["type"] == "Seizure"]
        by_segment = defaultdict(int)
        for e in seizures:
            by_segment[e["segment"]] += 1
        hit = sum(1 for s in short if by_segment.get(s.index))
        print("%-46s %5d %7d %8d" % (name, len(reader.segments), len(short), hit))

    print("\nusable signal by segment length (window budget):")
    segments = [s for reader, _ in per_file.values() for s in reader.segments]
    for limit in (4, 10, 30, 60):
        fit = [s for s in segments if s.duration_sec >= limit]
        hours = sum(s.duration_sec for s in fit) / 3600
        print("    windows of %2d s: %2d/%d segments, %.1f h"
              % (limit, len(fit), len(segments), hours))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
