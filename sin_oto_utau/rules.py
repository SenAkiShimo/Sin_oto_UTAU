from sin_oto_utau.oto_parser import OtoEntry


def fix_entry(entry: OtoEntry) -> OtoEntry:
    entry.offset = max(0.0, float(entry.offset))
    entry.consonant = max(1.0, float(entry.consonant))
    entry.preutterance = max(1.0, float(entry.preutterance))
    entry.overlap = max(0.0, float(entry.overlap))

    if entry.overlap > entry.preutterance * 0.6:
        entry.overlap = entry.preutterance * 0.6

    if entry.consonant < entry.preutterance:
        entry.consonant = entry.preutterance

    return entry