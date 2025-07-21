import json
from datetime import datetime

from nicegui import ui

from src.ui import GS, DATA_DIR
from src.ui.utils import ControlManager, CountryUtils, get_existing_sorted, MyUI

# --- UI Control Registry ---
CM = ControlManager()


######
# IO #
######


def _is_new():
    """Return True if current selected LICS is '<NEW>'."""
    return CM["select_lics"].value == "<NEW>"


def _check_save(_=None):
    """Trigger save validation."""
    name_exists = bool(CM["input_name"].value.strip())
    name_valid = CM["input_name"].validate()
    locality = CM["input_locality"].value.strip()

    is_valid = name_exists and name_valid and bool(locality)

    # Save button enable/disable
    CM.update("button_save", props="disable", props_remove=is_valid)

    # Warning text visibility & content
    if not is_valid:
        lines = []
        if not name_exists:
            lines.append("  • Session Name is required.")
        if not name_valid:
            lines.append("  • Session Name is invalid.")
        if not locality:
            lines.append("  • Locality is required.")
        CM.update("text_warn", "\n" + "\n\n".join(lines),
                  classes="hidden", classes_remove=True)
    else:
        CM.update("text_warn", classes="hidden")


def _save_lics(_=None):
    """Save current LICS and update selection."""
    name = CM["input_name"].value.strip()
    folder = DATA_DIR / name
    folder.mkdir(parents=True, exist_ok=True)
    json_path = folder / "ui_state.json"

    # --- Abort if file already exists ---
    if json_path.exists():
        ui.notify(f'LICS "{name}" already exists. Save aborted.', color='negative')
        return

    # --- Construct data dictionary ---
    data = {
        "name": name,
        "location": {
            "country": CM["select_country"].value,
            "subdivision": CM["select_subdivision"].value,
            "locality": CM["input_locality"].value.strip(),
        },
        "origin": {
            "lat": CM["number_lat"].value,
            "lon": CM["number_lon"].value,
            "heading": CM["number_heading"].value,
        },
        "notes": CM["input_notes"].value.strip(),
        "create_time": datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    }

    # --- Write to JSON file ---
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)  # noqa
        ui.notify(f'LICS saved to {json_path}.', color='positive')
    except Exception as e:  # noqa
        ui.notify(f'Failed to save {json_path}: {e}', color='negative')
        return

    # --- Add new LICS to dropdown options ---
    options = CM["select_lics"].options
    CM.update("select_lics", name,
              options=["<NEW>"] + [name] + options[1:])


def _load_lics(json_path, name=None):
    """Load LICS from json."""
    with open(json_path, "r") as f:
        data = json.load(f)
    if name is None:
        name = data["name"]
    CM.update("input_name", name)
    CM.update("select_country", data["location"]["country"])
    CM.update("select_subdivision", data["location"]["subdivision"],
              options=CountryUtils.get_subdivisions(data["location"]["country"]))
    CM.update("input_locality", data["location"]["locality"])
    CM.update("number_lat", data["origin"]["lat"])
    CM.update("number_lon", data["origin"]["lon"])
    CM.update("number_heading", data["origin"]["heading"])
    CM.update("input_notes", data["notes"])
    CM.update("input_time", data["create_time"])


def _restore_for_new():
    """Try restore from existing for new."""
    try:
        # First, try last selection
        name = CM['last_selection']
        json_path = DATA_DIR / f"{name}" / "ui_state.json"
        _load_lics(json_path, name="")
    except Exception as e:  # noqa
        try:
            # Second, try latest creation
            name = CM['select_lics'].options[1]
            json_path = DATA_DIR / f"{name}" / "ui_state.json"
            _load_lics(json_path, name="")
        except Exception as e:  # noqa
            # Finally, fall back to default
            CM.update("input_name", "")
            CM.update("select_country", CountryUtils.COUNTRIES_NAMES[0])
            CM.update("input_locality", "")
            CM.update("number_heading", 0.0)
            CM.update("input_notes", "")

    # For safety
    _check_save()


def _on_change_select_lics(_=None):
    """Handle user changing selected LICS, including loading saved LICS data."""
    is_new = _is_new()

    # 1. Set all field readonly states based on selection
    for key in [
        "input_name",
        "select_country", "select_subdivision", "input_locality",
        "number_lat", "number_lon", "number_heading",
        "input_notes"
    ]:
        CM.update(key, props="readonly", props_remove=is_new)

    # 2. Toggle visibility of controls
    CM.update("input_time", classes="hidden", classes_remove=not is_new)
    CM.update("button_save", classes="hidden", classes_remove=is_new)
    CM.update("text_warn", classes="hidden", classes_remove=is_new)

    # 3. If loading existing LICS, populate fields from saved data
    if not is_new:
        name = CM["select_lics"].value
        json_path = DATA_DIR / name / "ui_state.json"
        try:
            _load_lics(json_path)

            # Log the last valid selection
            CM["last_selection"] = name
        except Exception as e:  # noqa
            ui.notify(f'Failed to load/parse "{json_path}": {e}', color='negative')

            # Update options and selection
            lics_options = ["<NEW>"] + get_existing_sorted(DATA_DIR)
            CM.update("select_lics", "<NEW>", options=lics_options)
    else:
        # Load previous with fallback: last selection -> latest creation -> default
        _restore_for_new()


###################
# Local Interplay #
###################


def _on_change_select_country(_=None):
    """Handle update of country: update subdivisions and lat/lon."""
    country = CM["select_country"].value

    # --- Subdivisions ---
    sub_list = CountryUtils.get_subdivisions(country)
    CM.update("select_subdivision",
              options=sub_list,
              value=sub_list[0] if sub_list else "")

    # --- Lat/Lon ---
    lat, lon = CountryUtils.get_latlon(country)
    CM.update("number_lat", value=lat)
    CM.update("number_lon", value=lon)


###########################
# MAIN UI INITIALIZATION  #
###########################


def initialize():
    """Initialize the entire LICS tab UI."""
    # ui.label("Define LICS").classes('text-3xl font-bold')

    # --- Selection ---
    lics_options = ["<NEW>"] + get_existing_sorted(DATA_DIR)
    CM["select_lics"] = ui.select(
        lics_options,
        value="<NEW>",
        on_change=_on_change_select_lics,
        label="Select or Create a LICS"
    ).classes('w-full')
    CM["select_lics"].bind_value_to(GS, "selected_lics")

    # --- Name Input ---
    CM["input_name"] = ui.input(
        "Name",
        on_change=_check_save,
        validation={
            "Name cannot be '<NEW>'": lambda name: name.strip() != "<NEW>"
        }
    ).classes('w-full').props('autocomplete=off')

    # --- Location Row ---
    with MyUI.row():
        CM["select_country"] = ui.select(
            CountryUtils.COUNTRIES_NAMES,
            value=CountryUtils.COUNTRIES_NAMES[0],
            label="Country",
            on_change=_on_change_select_country
        ).classes('flex-1')

        sub_list = CountryUtils.get_subdivisions(CountryUtils.COUNTRIES_NAMES[0])
        CM["select_subdivision"] = ui.select(
            sub_list, label="Subdivision").classes('flex-1')
        CM.update("select_subdivision", value=sub_list[0] if sub_list else "",
                  options=sub_list)

        CM["input_locality"] = ui.input(
            "Locality", on_change=_check_save).classes('flex-1')

    # --- Origin Row ---
    lat, lon = CountryUtils.get_latlon(CountryUtils.COUNTRIES_NAMES[0])
    with MyUI.row():
        CM["number_lat"] = ui.number(
            "Latitude", value=lat,
            min=-90, max=90, step=0.01, format='%.2f'
        ).classes('flex-1')
        CM["number_lon"] = ui.number(
            "Longitude", value=lon,
            min=-180, max=180, step=0.01, format='%.2f'
        ).classes('flex-1')
        CM["number_heading"] = ui.number(
            "Heading", value=0.0,
            min=0, max=360, step=0.01, format='%.2f'
        ).classes('flex-1')

    # --- Notes Input ---
    CM["input_notes"] = ui.input("Notes").classes('w-full')

    # --- Create Time ---
    CM["input_time"] = ui.input("Create Time").classes('w-full hidden')
    CM.update("input_time", props="readonly")

    with ui.row().classes(f'w-full gap-10'):
        # --- Save Button ---
        CM["button_save"] = ui.button("Save LICS", color="primary",
                                      on_click=_save_lics) \
            .props("disable").classes('w-1/4 text-white font-semibold h-16')

        # --- Warning Box ---
        CM["text_warn"] = ui.textarea(
            label="⚠️ Saving disabled:",
            value="\n  • LICS Name is required.\n"
                  "\n  • Locality is required.",
        ).props('readonly borderless').classes('w-1/4 text-base large-label h-24')
