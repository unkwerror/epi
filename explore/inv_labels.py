"""Inventory step 4b: repair marker labels whose Cyrillic text arrives garbled.

The file stores strings as UTF-16LE, but neo decodes them one byte per
character. Cyrillic lives in U+04xx, so every letter turns into its low byte
followed by a stray '\\x04'. Re-encoding as latin-1 and decoding as UTF-16LE
gives the original Russian text back.
"""

import sys
from collections import Counter
from pathlib import Path

from epi import NicoletEReader


def mend(text):
    try:
        repaired = text.encode("latin-1").decode("utf-16-le")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text
    return repaired.rstrip("\x00").strip()


def looks_garbled(text):
    """A stray NUL or U+0004 byte is the fingerprint of a misdecoded UTF-16 string."""
    return bool(text) and ("\x04" in text or "\x00" in text)


def main(root):
    kinds = Counter()
    for path in sorted(Path(root).rglob("*.e")):
        for event in NicoletEReader(path).read_events():
            kinds[event["type"]] += 1

    print("%-26s %-36s %s" % ("raw label", "repaired", "count"))
    for kind, count in kinds.most_common():
        if kind in ("UNKNOWN", "") or not looks_garbled(kind):
            continue
        print("%-26r %-36r %d" % (kind, mend(kind), count))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
