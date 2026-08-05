"""Inventory step 6: group the unnamed markers by their raw type GUID.

Neither reader knows the names behind some event type GUIDs, so they all show
up as UNKNOWN. The GUID itself is in the file, and grouping by it turns one
undifferentiated pile into a handful of distinct marker classes. The 'user'
field says which workstation or reviewer created the marker, which separates
bedside marks from ones added during review.
"""

import sys
from collections import defaultdict
from pathlib import Path

from epi import NicoletEReader, learn_type_names


def collect(root):
    groups = defaultdict(lambda: {"n": 0, "names": set(), "users": set(),
                                  "durations": [], "files": set(),
                                  "annotations": set()})
    paths = sorted(Path(root).rglob("*.e"))
    type_names = learn_type_names(paths)
    for path in paths:
        for event in NicoletEReader(path, type_names=type_names).read_events():
            entry = groups[event["guid"]]
            entry["n"] += 1
            entry["names"].add(event["type"])
            entry["users"].add(event["user"] or "-")
            entry["durations"].append(event["duration_sec"])
            entry["files"].add(path.parent.name)
            if event["text"]:
                entry["annotations"].add(event["text"])
    return groups


def main(root):
    groups = collect(root)
    print("%d distinct event type GUIDs\n" % len(groups))
    print("%-40s %5s %6s %9s  %-22s %s"
          % ("GUID", "count", "files", "med_dur_s", "name", "users"))

    for guid, entry in sorted(groups.items(), key=lambda kv: -kv[1]["n"]):
        durations = sorted(entry["durations"])
        median = durations[len(durations) // 2]
        print("%-40s %5d %6d %9.1f  %-22s %s"
              % (guid, entry["n"], len(entry["files"]), median,
                 "/".join(sorted(entry["names"]))[:22],
                 ",".join(sorted(entry["users"]))[:24]))
        if entry["annotations"]:
            sample = sorted(entry["annotations"])[:3]
            print("        annotations: %s" % "; ".join(s[:40] for s in sample))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
