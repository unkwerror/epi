"""Inventory step 3: which channels exist, and what is common to every file.

Montages differ between recordings in count, order and naming, so a training
pipeline has to address channels by name and settle on an intersection that is
present everywhere.
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

from epi import NicoletEReader

# the 19 electrodes of the standard 10-20 layout
TEN_TWENTY = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz",
]


def main(root):
    readers = {}
    for path in sorted(Path(root).rglob("*.e")):
        readers[path.parent.name] = NicoletEReader(path)

    print("%-46s %5s %6s  %s" % ("recording", "chans", "rate", "extras beyond 10-20"))
    for name, reader in readers.items():
        extra = [c for c in reader.channels if c not in TEN_TWENTY]
        print("%-46s %5d %6g  %s" % (name, len(reader.channels),
                                     reader.sampling_rate, ", ".join(extra)))

    print("\nsampling rates: %s"
          % dict(Counter(r.sampling_rate for r in readers.values())))

    common = set.intersection(*(set(r.channels) for r in readers.values()))
    print("\npresent in all %d recordings (%d channels):" % (len(readers), len(common)))
    print("    " + ", ".join(c for c in TEN_TWENTY if c in common)
          + " | " + ", ".join(sorted(common - set(TEN_TWENTY))))

    missing = [c for c in TEN_TWENTY if c not in common]
    print("\n10-20 electrodes missing somewhere: %s" % (missing or "none"))

    print("\nchannel name frequency across recordings:")
    counts = Counter(c for r in readers.values() for c in r.channels)
    for name, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if count < len(readers):
            print("    %-14s %d/%d" % (name, count, len(readers)))

    print("\nchannels dropped by the reader (different sampling rate):")
    dropped = defaultdict(set)
    for name, reader in readers.items():
        for channel, rate in reader.excluded:
            dropped[(channel, rate)].add(name)
    for (channel, rate), owners in sorted(dropped.items()):
        print("    %-10s %5g Hz  in %d recordings" % (channel, rate, len(owners)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
