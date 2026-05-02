"""Normalize the FAA OMIn maintenance dataset into Case records.

Source: data/.../OMIn_dataset/data/FAA_data/Maintenance_Text_data_nona.csv
Output: artifacts/cases/cases.jsonl  (one normalized Case per line)

Key FAA columns we read:
    c5   case id              c61  fatal injuries
    c1   status flag (A/I)    c63  serious injuries
    c9   event date YYYYMMDD  c65  minor injuries
    c11  region               c41  damage flag
    c13  state                c77  accident type description
    c14  city                 c79  cause/factor 1
    c23  aircraft make        c81  cause/factor 2
    c24  aircraft model       c93  event description
    c105 weather (VFR/IFR)    c95  flight phase
    c107 weather factor       c109 light condition
    c101 flight purpose       c119 narrative
    c146 weight class         c148 aircraft class
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
from tqdm import tqdm

from .config import SETTINGS
from .schema import Case, CaseMetadata


def _clean(val) -> str:
    """Normalize whitespace, drop NaN/'nan' literals."""
    if val is None:
        return ""
    if isinstance(val, float) and math.isnan(val):
        return ""
    s = str(val).strip()
    if s.lower() == "nan":
        return ""
    return re.sub(r"\s+", " ", s)


def _to_int(val) -> int:
    s = _clean(val)
    if not s:
        return 0
    try:
        return int(float(s))
    except ValueError:
        return 0


def _parse_date(yyyymmdd: str) -> tuple[str | None, int | None]:
    s = _clean(yyyymmdd)
    if len(s) != 8 or not s.isdigit():
        return None, None
    y, m, d = s[0:4], s[4:6], s[6:8]
    try:
        return f"{y}-{m}-{d}", int(y)
    except ValueError:
        return None, None


def _derive_status(c1: str, case_id: str) -> str:
    """ACCIDENT vs INCIDENT. The c5 id ends in 'A' (accident) or 'I' (incident)."""
    flag = _clean(c1).upper()[:1] or _clean(case_id)[-1:].upper()
    return "ACCIDENT" if flag == "A" else "INCIDENT" if flag == "I" else "UNKNOWN"


def _derive_severity(damage: str, status: str) -> str:
    """Derive severity from FAA damage code (c41) + report status.

    c41 codes seen in OMIn: '1', '3', '9', 'F3', 'F9', 'XX', '99', ''.
    The 'F' prefix signals fatal involvement; '9' / '99' / 'XX' indicate
    destroyed or unknown-but-severe damage.
    """
    d = damage.upper().strip()
    if d.startswith("F"):
        return "CRITICAL"
    if d in {"9", "99", "XX"}:
        return "HIGH"
    if status == "ACCIDENT":
        return "MEDIUM"
    return "LOW"


def _derive_summary(row: dict) -> str:
    accident_type = _clean(row.get("c77"))
    city = _clean(row.get("c14"))
    state = _clean(row.get("c13"))
    make = _clean(row.get("c23"))
    model = _clean(row.get("c24"))
    aircraft = " ".join(p for p in [make, model] if p)
    location = ", ".join(p for p in [city, state] if p)
    head = accident_type or "Aviation maintenance event"
    parts = [head]
    if aircraft:
        parts.append(aircraft)
    if location:
        parts.append(location)
    return " | ".join(parts)


def _derive_comments(row: dict) -> list[str]:
    """Synthesize a comments stream from cause/event/condition columns."""
    fields = [
        ("Accident type", row.get("c77")),
        ("Primary factor", row.get("c79")),
        ("Person/role", row.get("c81")),
        ("Secondary factor", row.get("c83")),
        ("Tertiary factor", row.get("c85")),
        ("Event", row.get("c93")),
        ("Flight phase", row.get("c95")),
        ("Cause origin", row.get("c99")),
        ("Flight purpose", row.get("c101")),
        ("Operation type", row.get("c103")),
        ("Weather rules", row.get("c105")),
        ("Weather factor", row.get("c107")),
        ("Light condition", row.get("c109")),
    ]
    out = []
    for label, raw in fields:
        v = _clean(raw)
        if v:
            out.append(f"{label}: {v}")
    return out


def _row_to_case(row: dict) -> Case | None:
    case_id = _clean(row.get("c5"))
    description = _clean(row.get("c119"))
    if not case_id or not description:
        return None

    status = _derive_status(row.get("c1", ""), case_id)
    damage = _clean(row.get("c41"))
    severity = _derive_severity(damage, status)

    event_date, event_year = _parse_date(row.get("c9", ""))
    metadata = CaseMetadata(
        event_date=event_date,
        event_year=event_year,
        state=_clean(row.get("c13")) or None,
        city=_clean(row.get("c14")) or None,
        region=_clean(row.get("c11")) or None,
        aircraft_make=_clean(row.get("c23")) or None,
        aircraft_model=_clean(row.get("c24")) or None,
        aircraft_class=_clean(row.get("c148")) or None,
        weight_class=_clean(row.get("c146")) or None,
        weather=_clean(row.get("c105")) or None,
        light_condition=_clean(row.get("c109")) or None,
        flight_phase=_clean(row.get("c95")) or None,
        flight_purpose=_clean(row.get("c101")) or None,
    )

    return Case(
        case_id=case_id,
        summary=_derive_summary(row),
        description=description,
        status=status,
        comments=_derive_comments(row),
        severity=severity,
        metadata=metadata,
    )


def iter_cases(csv_path: Path | None = None) -> Iterable[Case]:
    csv_path = csv_path or SETTINGS.faa_csv
    if not csv_path.exists():
        raise FileNotFoundError(f"FAA CSV not found at {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False, dtype=str, keep_default_na=False)
    for _, row in df.iterrows():
        case = _row_to_case(row.to_dict())
        if case is not None:
            yield case


def build_cases(out_path: Path | None = None, limit: int | None = None) -> int:
    out_path = out_path or (SETTINGS.cases_dir / "cases.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for case in tqdm(iter_cases(), desc="normalizing cases"):
            f.write(case.model_dump_json() + "\n")
            count += 1
            if limit and count >= limit:
                break
    return count


def load_cases(path: Path | None = None) -> list[Case]:
    path = path or (SETTINGS.cases_dir / "cases.jsonl")
    cases = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(Case.model_validate_json(line))
    return cases


if __name__ == "__main__":
    n = build_cases()
    print(f"Wrote {n} cases to {SETTINGS.cases_dir / 'cases.jsonl'}")
