from pathlib import Path
from dataclasses import dataclass


@dataclass
class OtoEntry:
    wav: str
    alias: str
    offset: float
    consonant: float
    cutoff: float
    preutterance: float
    overlap: float


def parse_oto_line(line: str) -> OtoEntry | None:
    line = line.strip()

    if not line or "=" not in line:
        return None

    wav, rest = line.split("=", 1)
    parts = rest.split(",")

    if len(parts) != 6:
        return None

    try:
        return OtoEntry(
            wav=wav.strip(),
            alias=parts[0].strip(),
            offset=float(parts[1]),
            consonant=float(parts[2]),
            cutoff=float(parts[3]),
            preutterance=float(parts[4]),
            overlap=float(parts[5]),
        )
    except ValueError:
        return None


def read_oto(path: str | Path) -> list[OtoEntry]:
    path = Path(path)
    entries: list[OtoEntry] = []

    with path.open("r", encoding="utf-8-sig", errors="ignore") as f:
        for line in f:
            entry = parse_oto_line(line)
            if entry is not None:
                entries.append(entry)

    return entries


def write_oto(entries: list[OtoEntry], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for e in entries:
            line = (
                f"{e.wav}={e.alias},"
                f"{e.offset:.3f},"
                f"{e.consonant:.3f},"
                f"{e.cutoff:.3f},"
                f"{e.preutterance:.3f},"
                f"{e.overlap:.3f}\n"
            )
            f.write(line)