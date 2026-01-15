import requests

TIMEOUT = 15

# Minimal controlled labels (you can extend if you want)
WRB_ALLOWED = {
    "Acrisols","Alisols","Andosols","Anthrosols","Arenosols","Calcisols","Cambisols","Chernozems",
    "Cryosols","Durisols","Ferralsols","Fluvisols","Gleysols","Gypsisols","Histosols","Kastanozems",
    "Leptosols","Lixisols","Luvisols","Nitisols","Phaeozems","Planosols","Plinthosols","Podzols",
    "Regosols","Retisols","Solonchaks","Solonetz","Stagnosols","Technosols","Umbrisols","Vertisols"
}

TEXTURE_LABELS = {
    "Sand","Loamy Sand","Sandy Loam","Loam","Silt Loam","Silt",
    "Sandy Clay Loam","Clay Loam","Silty Clay Loam",
    "Sandy Clay","Silty Clay","Clay",
    "Unknown",
}


def _get_json(url, params):
    r = requests.get(url, params=params, timeout=TIMEOUT, headers={"Accept": "application/json"})
    r.raise_for_status()
    return r.json()


def _pick_topsoil_value(layer_dict):
    """
    layer_dict: a single element of properties.layers (e.g. name='sand')
    Return (value, d_factor) from 0-5cm using mean, else Q0.5, else None.
    """
    if not isinstance(layer_dict, dict):
        return None, None

    d_factor = None
    um = layer_dict.get("unit_measure")
    if isinstance(um, dict):
        d_factor = um.get("d_factor")

    depths = layer_dict.get("depths")
    if not isinstance(depths, list) or not depths:
        return None, d_factor

    # pick 0-5cm if present, else first
    entry = None
    for e in depths:
        if not isinstance(e, dict):
            continue
        rng = e.get("range") or {}
        if rng.get("top_depth") == 0 and rng.get("bottom_depth") == 5:
            entry = e
            break
    if entry is None:
        entry = depths[0] if isinstance(depths[0], dict) else None
    if entry is None:
        return None, d_factor

    values = entry.get("values")
    if not isinstance(values, dict):
        return None, d_factor

    v = values.get("mean")
    if v is None:
        v = values.get("Q0.5")
    return v, d_factor


def _to_percent(v, d_factor):
    """
    SoilGrids here returns g/kg with d_factor=10 (so % = v / 10).
    If d_factor missing, fall back to heuristic.
    """
    if v is None:
        return None
    try:
        v = float(v)
    except Exception:
        return None

    if isinstance(d_factor, (int, float)) and d_factor not in (0, 0.0):
        return v / float(d_factor)

    # fallback heuristic
    if v > 100:
        return v / 10.0
    if 0.0 <= v <= 1.0:
        return v * 100.0
    return v


def _fetch_texture_fractions(lat, lon):
    """
    Return sand,silt,clay in % (topsoil), or (None,None,None) on failure.
    """
    base = "https://rest.isric.org/soilgrids/v2.0/properties/query"

    def one(prop):
        j = _get_json(base, {"lat": float(lat), "lon": float(lon), "property": prop})
        layers = ((j.get("properties") or {}).get("layers") or [])
        layer = None
        for ly in layers:
            if isinstance(ly, dict) and ly.get("name") == prop:
                layer = ly
                break
        if layer is None and layers and isinstance(layers[0], dict):
            layer = layers[0]

        raw, d_factor = _pick_topsoil_value(layer)
        return _to_percent(raw, d_factor)

    try:
        sand = one("sand")
        silt = one("silt")
        clay = one("clay")
        return sand, silt, clay
    except Exception:
        return None, None, None


def _fetch_wrb_group(lat, lon):
    """
    Return WRB group string or None.
    """
    url = "https://rest.isric.org/soilgrids/v2.0/classification/query"
    try:
        j = _get_json(url, {"lat": float(lat), "lon": float(lon), "number_classes": 1})
        name = None
        if isinstance(j, dict):
            for k in ("wrb_class_name", "class_name", "name"):
                if isinstance(j.get(k), str):
                    name = j.get(k)
                    break
        if not name:
            return None

        # light normalize: first token, plural as-is
        n = name.strip().replace("WRB", "").strip()
        first = n.split()[0].strip(",;()[]")
        if first in WRB_ALLOWED:
            return first
        if n in WRB_ALLOWED:
            return n
        return None
    except Exception:
        return None


def _usda_texture(sand, silt, clay):
    """
    Good-enough USDA texture classification.
    Inputs: % values (not necessarily summing to 100; will normalize).
    """
    if sand is None or silt is None or clay is None:
        return "Unknown"

    total = sand + silt + clay
    if total <= 0:
        return "Unknown"

    sand = sand * 100.0 / total
    silt = silt * 100.0 / total
    clay = clay * 100.0 / total

    # Simple rule set (coarse but consistent)
    if sand >= 85 and clay < 10 and silt < 15:
        return "Sand"
    if sand >= 70 and clay < 15 and silt < 30:
        return "Loamy Sand"
    if sand >= 52 and clay < 20 and silt < 50:
        return "Sandy Loam"

    if silt >= 80 and clay < 12:
        return "Silt"
    if silt >= 50 and clay < 27 and sand <= 50:
        return "Silt Loam"

    if clay >= 40 and silt <= 40 and sand <= 45:
        return "Clay"
    if clay >= 40 and sand > 45:
        return "Sandy Clay"
    if clay >= 40 and silt > 40:
        return "Silty Clay"

    if 27 <= clay < 40 and sand > 45:
        return "Sandy Clay Loam"
    if 27 <= clay < 40 and silt > 40:
        return "Silty Clay Loam"
    if 27 <= clay < 40:
        return "Clay Loam"

    # remaining mid-range
    if clay < 12 and sand < 70 and silt < 50:
        return "Loam"
    if 7 <= clay < 27 and 28 <= silt < 50 and sand < 52:
        return "Loam"

    return "Unknown"


def get_soil_summary(lat, lon):
    """
    Minimal public function.

    Returns:
      {"wrb_group": <WRB or 'Unknown'>, "texture_class": <Texture or 'Unknown'>}
    """
    wrb = _fetch_wrb_group(lat, lon) or "Unknown"

    sand, silt, clay = _fetch_texture_fractions(lat, lon)
    texture = _usda_texture(sand, silt, clay)
    if texture not in TEXTURE_LABELS:
        texture = "Unknown"

    return {"wrb_group": wrb, "texture_class": texture}


if __name__ == "__main__":
    # A point you verified works for sand:
    lat, lon = 42.0, -93.5
    print(get_soil_summary(lat, lon))
