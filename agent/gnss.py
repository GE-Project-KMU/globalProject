"""NMEA validation and UTC association. HDOP is never converted into meters."""
from dataclasses import dataclass
from datetime import datetime, timezone
import math


def _number(value, *, nonnegative=False):
    number = float(value)
    if not math.isfinite(number) or (nonnegative and number < 0):
        raise ValueError("invalid_nmea_numeric_value")
    return number


def _coordinate(raw, direction):
    if not raw or direction not in ("N", "S", "E", "W"):
        raise ValueError("invalid_coordinate")
    value = float(raw)
    degrees = int(value // 100)
    minutes = value - degrees * 100
    if not 0 <= minutes < 60:
        raise ValueError("invalid_coordinate_minutes")
    answer = degrees + minutes / 60
    maximum = 90 if direction in ("N", "S") else 180
    if not math.isfinite(answer) or answer > maximum:
        raise ValueError("coordinate_out_of_range")
    return -answer if direction in ("S", "W") else answer


def parse_nmea(sentence, received_at=None):
    received_at = received_at or datetime.now(timezone.utc).isoformat()
    received = datetime.fromisoformat(received_at.replace("Z", "+00:00"))
    if received.tzinfo is None:
        raise ValueError("timezone_required")
    sentence = sentence.strip()
    if not sentence.startswith("$") or "*" not in sentence:
        raise ValueError("missing_nmea_checksum")
    content, checksum = sentence[1:].split("*", 1)
    computed = 0
    for char in content:
        computed ^= ord(char)
    if checksum.upper() != f"{computed:02X}":
        raise ValueError("invalid_nmea_checksum")
    fields = content.split(",")
    kind = fields[0][-3:]
    result = {"kind": kind, "received_at": received_at}
    try:
        if kind == "RMC":
            if fields[2] != "A":
                return result | {"fix_valid": False, "lat": None, "lon": None}
            clock, date = fields[1], fields[9]
            if len(clock) < 6 or len(date) != 6:
                raise ValueError("invalid_nmea_time")
            moment = datetime.strptime(date + clock[:6], "%d%m%y%H%M%S").replace(tzinfo=timezone.utc)
            fraction = float("0." + clock.split(".")[1]) if "." in clock else 0
            moment = moment.replace(microsecond=round(fraction * 1_000_000))
            return result | {"fix_valid": True, "lat": _coordinate(fields[3], fields[4]),
                             "lon": _coordinate(fields[5], fields[6]), "utc": moment.isoformat(),
                             "speed_kmh": _number(fields[7] or 0, nonnegative=True) * 1.852, "course_deg": _number(fields[8]) if fields[8] else None}
        if kind == "GGA":
            fixed = int(fields[6]) > 0
            return result | {"fix_valid": fixed, "lat": _coordinate(fields[2], fields[3]) if fixed else None,
                             "lon": _coordinate(fields[4], fields[5]) if fixed else None,
                             "hdop": _number(fields[8], nonnegative=True) if fields[8] else None, "satellites": int(fields[7] or 0)}
        return result
    except (IndexError, OverflowError) as exc:
        raise ValueError("malformed_nmea") from exc


def unavailable(reason="missing_gnss"):
    return {"lat": None, "lon": None, "source": "vehicle_observation", "quality": "unavailable",
            "hdop": None, "age_ms": None, "accuracy_m": None}


def associate_location(captured_at, samples, max_age_seconds=2):
    at = datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
    if at.tzinfo is None:
        raise ValueError("timezone_required")
    usable = []
    for row in samples:
        if not row.get("fix_valid") or not row.get("utc"):
            continue
        try:
            when = datetime.fromisoformat(row["utc"].replace("Z", "+00:00"))
            lat, lon = float(row["lat"]), float(row["lon"])
            if not math.isfinite(lat) or not math.isfinite(lon) or not -90 <= lat <= 90 or not -180 <= lon <= 180:
                continue
        except (KeyError, TypeError, ValueError):
            continue
        if when.tzinfo is None:
            continue
        usable.append((abs((when - at).total_seconds()), row))
    if not usable:
        return unavailable()
    age, selected = min(usable, key=lambda x: x[0])
    if age > max_age_seconds:
        return unavailable("gnss_gap_over_2_seconds")
    hdop = selected.get("hdop")
    quality = "valid" if age <= .5 and hdop is not None and 0 <= hdop <= 3 else "degraded"
    return {"lat": selected["lat"], "lon": selected["lon"], "source": "vehicle_observation", "quality": quality,
            "hdop": hdop, "age_ms": round(age * 1000), "accuracy_m": selected.get("accuracy_m")}


def build_fixes(records):
    """Associate RMC UTC with nearby GGA HDOP, preserving parse diagnostics."""
    parsed, errors = [], []
    for index, row in enumerate(records):
        try:
            parsed.append(parse_nmea(row["sentence"], row.get("received_at")))
        except (ValueError, KeyError) as exc:
            errors.append({"index": index, "error": str(exc)})
    ggas = [row for row in parsed if row.get("kind") == "GGA"]
    fixes = []
    for row in parsed:
        if row.get("kind") != "RMC" or not row.get("utc"):
            continue
        at = datetime.fromisoformat(row["received_at"].replace("Z", "+00:00"))
        if at.tzinfo is None:
            errors.append({"error": "timezone_required"})
            continue
        candidates = []
        for gga in ggas:
            received = datetime.fromisoformat(gga["received_at"].replace("Z", "+00:00"))
            if received.tzinfo is None:
                continue
            age = abs((received - at).total_seconds())
            if age <= 1 and gga.get("fix_valid"):
                candidates.append((age, gga))
        best = min(candidates, default=None, key=lambda item: item[0])
        fixes.append(row | {"hdop": best[1].get("hdop") if best else None})
    return {"fixes": fixes, "errors": errors}
