"""Structured access to the lab reference-range CSV's critical thresholds.

Separate from `app.knowledge.loaders.load_lab_reference_range_documents`,
which turns the same CSV into text documents for semantic retrieval/citation.
Red-flag detection needs an exact number for a known `test_code`, not a
similarity search that might return the wrong test's passage — this reads
the CSV directly as structured data instead.
"""

import csv
from pathlib import Path

from pydantic import BaseModel

from app.knowledge.ingest import DATA_DIR


class CriticalThreshold(BaseModel):
    test_code: str
    critical_low: float | None = None
    critical_high: float | None = None


def load_critical_thresholds(path: Path | None = None) -> dict[str, CriticalThreshold]:
    csv_path = path or (DATA_DIR / "lab_reference_ranges.csv")
    thresholds: dict[str, CriticalThreshold] = {}
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            thresholds[row["test_code"]] = CriticalThreshold(
                test_code=row["test_code"],
                critical_low=float(row["critical_low"]) if row.get("critical_low") else None,
                critical_high=float(row["critical_high"]) if row.get("critical_high") else None,
            )
    return thresholds
