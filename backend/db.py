"""
NBIS SQLite accessor
====================
Thin read-only wrapper around the `nbis.db` file produced by the notebook's
"Egyptian Database" section. Runs the same JOIN across
`subjects / fathers / mothers / hospitals` that the notebook uses to print
a matched record.

Design notes
------------
- Opens a short-lived connection per call. SQLite handles this cheaply
  (the DB file stays mmap-ed by the OS) and avoids threading pitfalls with
  FastAPI's concurrent request handling.
- Returns `None` if the file is missing, the subject isn't found, or the
  schema differs — so the API degrades gracefully when the DB is optional.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

log = logging.getLogger("nbis.db")

_FULL_RECORD_SQL = """
    SELECT
        s.subject_id,
        s.full_name          AS child_name,
        s.gender             AS child_gender,
        s.birth_datetime,
        s.weight_kg,
        s.blood_group        AS child_blood,
        s.registration_date,

        f.father_id,
        f.full_name          AS father_name,
        f.national_id        AS father_nid,
        f.birth_date         AS father_dob,
        f.blood_group        AS father_blood,
        f.phone              AS father_phone,
        f.email              AS father_email,
        f.city               AS father_city,

        m.mother_id,
        m.full_name          AS mother_name,
        m.national_id        AS mother_nid,
        m.birth_date         AS mother_dob,
        m.blood_group        AS mother_blood,
        m.phone              AS mother_phone,
        m.email              AS mother_email,
        m.city               AS mother_city,

        h.hospital_id,
        h.hospital_name,
        h.city               AS hospital_city,
        h.phone              AS hospital_phone,
        h.address            AS hospital_address
    FROM subjects  s
    JOIN fathers   f ON s.father_id   = f.father_id
    JOIN mothers   m ON s.mother_id   = m.mother_id
    JOIN hospitals h ON s.hospital_id = h.hospital_id
    WHERE s.subject_id = ?
    LIMIT 1
"""


class NBISDatabase:
    """Read-only accessor around `nbis.db`."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)

    # ─────────────────────────────────────────────────────────────────
    @property
    def available(self) -> bool:
        return self.db_path.exists() and self.db_path.is_file()

    # ─────────────────────────────────────────────────────────────────
    def fetch_full_record(self, subject_id: str) -> dict[str, Any] | None:
        """
        Return a fully-joined dict for the given subject_id.

        Shape (when MATCH):
            {
              "child":    { name, gender, birth_datetime, weight_kg, blood_group, ... },
              "father":   { name, national_id, birth_date, blood_group, phone, email, city },
              "mother":   { name, national_id, birth_date, blood_group, phone, email, city },
              "hospital": { name, city, phone, address }
            }
        Returns None if the DB is missing or no matching subject row exists.
        """
        if not self.available:
            return None

        try:
            # read-only URI opens the file without creating a new one if missing
            uri = f"file:{self.db_path.as_posix()}?mode=ro"
            with sqlite3.connect(uri, uri=True) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(_FULL_RECORD_SQL, (str(subject_id),)).fetchone()
        except sqlite3.Error as e:
            log.warning("SQLite error for subject_id=%s: %s", subject_id, e)
            return None

        if row is None:
            return None

        r = dict(row)
        return {
            "child": {
                "subject_id"       : r["subject_id"],
                "name"             : r["child_name"],
                "gender"           : r["child_gender"],
                "birth_datetime"   : r["birth_datetime"],
                "weight_kg"        : r["weight_kg"],
                "blood_group"      : r["child_blood"],
                "registration_date": r.get("registration_date"),
            },
            "father": {
                "father_id"  : r["father_id"],
                "name"       : r["father_name"],
                "national_id": r["father_nid"],
                "birth_date" : r["father_dob"],
                "blood_group": r["father_blood"],
                "phone"      : r["father_phone"],
                "email"      : r["father_email"],
                "city"       : r["father_city"],
            },
            "mother": {
                "mother_id"  : r["mother_id"],
                "name"       : r["mother_name"],
                "national_id": r["mother_nid"],
                "birth_date" : r["mother_dob"],
                "blood_group": r["mother_blood"],
                "phone"      : r["mother_phone"],
                "email"      : r["mother_email"],
                "city"       : r["mother_city"],
            },
            "hospital": {
                "hospital_id": r["hospital_id"],
                "name"       : r["hospital_name"],
                "city"       : r["hospital_city"],
                "phone"      : r["hospital_phone"],
                "address"    : r["hospital_address"],
            },
        }
