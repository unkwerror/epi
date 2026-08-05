"""Inventory step 1: who are the patients behind the .e files.

Only the file header is parsed, so this is cheap even for gigabyte recordings.
Files are grouped by the patientID GUID stored inside the header, because the
folder names carry export numbers rather than a stable patient identity.
"""

import sys
from collections import defaultdict
from pathlib import Path

from epi import NicoletEReader


def main(root):
    by_patient = defaultdict(list)
    for path in sorted(Path(root).rglob("*.e")):
        reader = NicoletEReader(path, type_names={})
        by_patient[reader.patient_key].append((path, reader.patient))

    n_files = sum(len(v) for v in by_patient.values())
    missing = sum(1 for k in by_patient if k.startswith("altid:"))
    print("unique patients: %d over %d files (%d without a patientID GUID)\n"
          % (len(by_patient), n_files, missing))

    for key, entries in sorted(by_patient.items(), key=lambda kv: -len(kv[1])):
        info = entries[0][1]
        print("%-38s alt_id=%-7s dob=%s  sex=%s  recordings=%d"
              % (key, info["alt_id"], info["dob"],
                 (info["sex_id"] or "")[1:9], len(entries)))
        if info["notes"]:
            print("    notes: %s" % info["notes"][:110])
        for path, entry in entries:
            print("    export %-3s %s/%s"
                  % (entry["last_name"], path.parent.parent.name, path.parent.name))
        print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
