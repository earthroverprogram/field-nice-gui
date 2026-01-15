"""
Build lat/lon index for ALL pycountry subdivisions using OpenStreetMap Nominatim.

Upgraded:
- Save after EACH successful fetch (checkpointing)
- Resume from existing JSON; skip already fetched subdivision codes
- Atomic writes + .bak recovery
- Optional error log

Requirements:
    pip install pycountry requests numpy
"""

import json
import re
import time
from pathlib import Path
from typing import Optional

import pycountry
import requests
from countryinfo import CountryInfo

# -------------------- CONFIG --------------------

OUTPUT_FILE = Path("static_data/pycountry_subdivision_latlon.json")
ERROR_LOG = Path("static_data/pycountry_subdivision_latlon.errors.log")

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "subdivision-latlon-builder"

REQUEST_DELAY_SEC = 0.5
REQUEST_TIMEOUT = 15

# If True, also persist "not found" entries to avoid retrying forever.
# You can set to False if you prefer to retry them in later runs.
PERSIST_NOT_FOUND = True


# -----------------------------------------------

def _country_latlon(country_name: str) -> tuple[float | None, float | None]:
    try:
        info = CountryInfo(country_name).info()
        latlng = info.get("latlng")
        if latlng and len(latlng) == 2:
            return float(latlng[0]), float(latlng[1])
    except Exception:
        pass
    return None, None


def _remove_brackets(name: str) -> str:
    """
    Remove any content wrapped by () or [] (including the brackets themselves)
    and normalize whitespace.
    """
    if not name:
        return name

    # Remove (...) and [...]
    cleaned = re.sub(r"\s*[\(\[].*?[\)\]]\s*", " ", name)

    # Normalize spaces
    return " ".join(cleaned.split())


def _load_existing(path: Path) -> dict[str, dict]:
    """Load existing JSON if present; attempt .bak recovery if needed."""
    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        bak = path.with_suffix(path.suffix + ".bak")
        if bak.exists():
            with open(bak, "r", encoding="utf-8") as f:
                return json.load(f)
        # If both fail, start fresh but keep the broken file for inspection.
        broken = path.with_suffix(path.suffix + ".broken")
        try:
            path.rename(broken)
        except Exception:
            pass
        return {}


def _atomic_write_json(path: Path, data: dict) -> None:
    """Atomic JSON write with backup to prevent corruption on crash."""
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + ".tmp")
    bak = path.with_suffix(path.suffix + ".bak")

    payload = json.dumps(data, indent=2, ensure_ascii=False)

    # Write tmp
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(payload)
        f.flush()

    # Rotate: current -> bak (best-effort)
    if path.exists():
        try:
            path.replace(bak)
        except Exception:
            pass

    # tmp -> current
    tmp.replace(path)


def _append_error(line: str) -> None:
    ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


def query_latlon(query: str) -> Optional[tuple[float, float]]:
    """Query Nominatim for a place centroid."""
    params = {
        "q": query,
        "format": "json",
        "limit": 1,
    }
    headers = {"User-Agent": USER_AGENT}

    r = requests.get(
        NOMINATIM_URL,
        params=params,
        headers=headers,
        timeout=REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    data = r.json()

    if not data:
        return None

    return float(data[0]["lat"]), float(data[0]["lon"])


def main():
    # Resume from existing progress
    result: dict[str, dict] = _load_existing(OUTPUT_FILE)
    done_codes = {
        k for k, v in result.items()
        if v.get("lat") is not None and v.get("lon") is not None
    }

    # alpha-2 -> country name
    country_name_map = {
        c.alpha_2: c.name
        for c in pycountry.countries
        if hasattr(c, "alpha_2")
    }

    subs = list(pycountry.subdivisions)
    total = len(subs)

    # Count how many from pycountry are already done (some saved codes may not exist anymore, that's fine)
    already = sum(1 for s in subs if s.code in done_codes)
    print(f"Total subdivisions: {total}")
    print(f"Already in output: {len(done_codes)} entries; matches current pycountry: {already}/{total}")

    for i, sub in enumerate(subs, start=1):
        if sub.code in done_codes:
            # Resume: skip completed
            continue

        country_name = country_name_map.get(sub.country_code)
        if not country_name:
            # Not expected, but skip safely
            continue

        country_name_query = country_name
        if "Republic of" in country_name_query:
            a, b = country_name_query.split(",")
            country_name_query = b.strip() + " " + a.strip()
        query = f"{_remove_brackets(sub.name)}, {_remove_brackets(country_name_query)}"

        try:
            latlon = query_latlon(query)

            if latlon is None:
                print(f"[{i}/{total}] SKIP (not found): {sub.code} | {query}")
                if PERSIST_NOT_FOUND:
                    # Persist sentinel so we won't retry forever
                    clat, clon = _country_latlon(country_name)
                    result[sub.code] = {
                        "country": country_name,
                        "subdivision": sub.name,
                        "lat": clat,
                        "lon": clon
                    }
                    _atomic_write_json(OUTPUT_FILE, result)
                    done_codes.add(sub.code)
                time.sleep(REQUEST_DELAY_SEC)
                continue

            lat, lon = latlon
            result[sub.code] = {
                "country": country_name,
                "subdivision": sub.name,
                "lat": lat,
                "lon": lon,
            }

            # ✅ checkpoint after each successful fetch
            _atomic_write_json(OUTPUT_FILE, result)
            done_codes.add(sub.code)

            print(f"[{i}/{total}] OK: {sub.code} -> ({lat:.4f}, {lon:.4f}) | saved")

        except Exception as e:
            msg = f"[{i}/{total}] ERROR: {sub.code} | {query} | {repr(e)}"
            print(msg)
            _append_error(msg)

        time.sleep(REQUEST_DELAY_SEC)

    print(f"\nSaved {len(result)} entries to {OUTPUT_FILE.resolve()}")
    print(f"Error log: {ERROR_LOG.resolve()}")


if __name__ == "__main__":
    main()
