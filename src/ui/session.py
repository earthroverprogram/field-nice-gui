import json
import re
import time
from datetime import datetime
from types import SimpleNamespace

import numpy as np
from nicegui import ui

from src.device.datalogger import Datalogger
from src.ui import GS, DATA_DIR
from src.ui.utils import ControlManager, get_existing_sorted, MyPlot, MyUI, CallbackBlocker

# --- UI Control Registry ---
CM = ControlManager()

# --- UI Callback Blocker (Batch Assigner) ---
CBB = CallbackBlocker()

############
# Defaults #
############


# Options
with open("src/ui/defaults/session_options.json", "r", encoding="utf-8") as f:
    SESSION_OPTIONS = json.load(f)

# Default layout code
DEFAULT_LAYOUT_CODE = '''def custom_layout():
    """Python function to generate user-defined layout."""
    import numpy as np  # Must import numpy within function
    theta = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    x = np.cos(theta) * 100
    y = np.sin(theta) * 100
    return np.stack([x, y], axis=1)  # Must return an (N, 2) array
'''


##########
# Layout #
##########


def _compute_layout(plus_shift=True):
    """Compute layout from Method."""
    grid = CM["select_layout"].value.upper() == "GRID-1D/2D"
    layout = None
    if grid:
        # Read grid parameters
        nx = round(CM["number_nx"].value)
        ny = round(CM["number_ny"].value)
        ox = CM["number_ox"].value
        oy = CM["number_oy"].value
        dx = CM["number_dx"].value
        dy = CM["number_dy"].value
        sx = CM["number_sx"].value
        sy = CM["number_sy"].value

        # Compute grid layout
        layout = []
        for iy in range(ny):
            for ix in range(nx):
                x = ox + ix * dx + iy * sx
                y = oy + iy * dy + ix * sy
                layout.append((x, y))
        layout = np.array(layout)
    else:
        # Run user code safely
        try:
            user_code = CM["code_custom"].value
            local_env = {}
            exec(user_code, {}, local_env)
            layout = local_env["custom_layout"]()
            if (not isinstance(layout, np.ndarray) or
                    layout.ndim != 2 or layout.shape[1] != 2 or layout.shape[0] == 0):
                raise ValueError("Function custom_layout() must return a "
                                 "numpy array of shape [N, 2], with N > 0.")
        except Exception as e:  # noqa
            ui.notify(f"Error in custom code: {e}", color="negative")

    if layout is not None:
        if plus_shift:
            try:
                for ch, shift in CM["shift_layout"].items():
                    layout[ch - 1] += shift
            except Exception as e:  # noqa
                ui.notify(f"Error applying layout shift : {e}", color="negative")
        layout = np.round(layout, decimals=0).astype(int)  # cm is minimum unit

        # --- Check for overlapping channels ---
        if layout.shape[0] > 1:
            # Compute pairwise distances between all layout points
            dist_matrix = np.linalg.norm(layout[:, None, :] - layout[None, :, :], axis=-1)
            np.fill_diagonal(dist_matrix, np.inf)  # ignore self-distance

            coincide_pairs = np.argwhere(np.isclose(dist_matrix, 0.0))
            if coincide_pairs.size > 0:
                # Only show each pair once (i < j)
                unique_pairs = [f"{i + 1} & {j + 1}" for i, j in coincide_pairs if i < j]
                if unique_pairs:
                    ui.notify(
                        f"Channel overlap detected between: [{', '.join(unique_pairs)}]",
                        color="warning"
                    )
    return layout


def _plot_layout(fig, layout):
    """Plot a layout."""
    fig.clear()

    if layout is None:
        MyPlot.error(fig, "Error Layout")
        return

    # Plot scatters
    ax = fig.gca()
    ax.grid(True, color=MyUI.gray_color(), linewidth=0.5)
    ax.scatter(layout[:, 0], layout[:, 1], s=50, color=MyUI.primary_color(), zorder=10)
    MyPlot.apply_margin(fig, layout)
    fig.tight_layout(pad=0)
    MyPlot.apply_dark(fig)


def _refresh_layout():
    """Update layout from scratch."""
    layout = _compute_layout()
    if layout is None:
        # Expansion
        CM["expansion_layout"].text = "⚠️ Error Layout"

        # Table
        CM["table_layout"].rows = []
        CM["table_layout"].update()
        CM["table_layout"].style(f"background-color: {MyUI.bg_color(True)}")

        # Flag
        CM["is_layout_valid"] = False
    else:
        # Expansion
        CM["expansion_layout"].text = \
            f"Layout Details: 【 {len(layout)} Channels 】"

        # Table
        CM["table_layout"].clear()
        new_rows = []
        for idx, (x, y) in enumerate(layout):
            # Naming
            ch = idx + 1
            default = CM["input_naming"].value.format(CH=ch)
            naming = CM["overwrite_naming"].get(ch, default)
            if not is_valid_naming(naming):
                ui.notify("Impossible Error. Report for debugging.", color="negative")
                naming = "E.RR.OR"

            # Shift
            shift = CM["shift_layout"].get(ch, (0, 0))
            dx, dy = shift

            # Row
            new_rows.append({
                'channel': ch,
                'x': x,
                'y': y,
                'dx': dx,
                'dy': dy,
                'naming': naming,
                'lock_shift': not _is_new()
            })
        CM["table_layout"].rows = new_rows
        CM["table_layout"].update()
        CM["table_layout"].style("background-color: var(--q-surface)")

        # Flag
        CM["is_layout_valid"] = True

    # Figure
    with CM["figure_layout"]:
        _plot_layout(CM["figure_layout"], layout)

    # Check save
    _check_save()


def _reset_shift_trailing():
    """Reset shift of trailing."""
    layout = _compute_layout(plus_shift=False)

    # Clear shift
    CM["shift_layout"] = {}

    if layout is not None:
        # Reset trailing channel
        min_ch = len(layout) + 1
        CM["number_st_ch"].min = min_ch  # Must go first
        if CM["number_st_ch"] and CM["number_st_ch"].value < min_ch:
            CM.update("number_st_ch", min_ch)


def _on_change_layout_params(_=None):
    """Handle user update of layout fields and trigger validation."""
    if CBB.blocking():
        return
    if _is_new():
        _reset_shift_trailing()
    _refresh_layout()
    _check_channels()


def _on_change_select_layout(_=None):
    """Handle user update of layout method and trigger validation."""
    use_grid = CM["select_layout"].value.upper() == "GRID-1D/2D"
    CM.update("container_grid", classes="hidden", classes_remove=use_grid)
    CM.update("code_custom", classes="hidden", classes_remove=not use_grid)
    _on_change_layout_params()


def _on_change_shift(e):
    """Handle editing shift layout in table."""
    row = e.args  # props.row
    ch = row['channel']
    try:
        dx = int(row['dx'])
        dy = int(row['dy'])
    except:  # noqa:
        return  # Do nothing, wait user to finish
    if dx == 0 and dy == 0:
        CM["shift_layout"].pop(ch, None)
    else:
        CM["shift_layout"][ch] = (dx, dy)
    _refresh_layout()


def _on_blur_shift(e):
    """Handle blur shift layout in table."""
    row = e.args  # props.row
    ch = row['channel']
    try:
        _ = int(row['dx'])
        _ = int(row['dy'])
        # Do nothing, changes already honored by _on_change_shift
    except:  # noqa:
        CM["shift_layout"].pop(ch, None)
        _refresh_layout()


def is_valid_naming(text: str) -> bool:
    """Name format validation: A.B.C with no spaces and valid template."""
    if not re.fullmatch(r'\S+\.\S+\.\S+', text):
        return False
    try:
        text.format(CH=1)
        return True
    except:  # noqa
        return False


def _on_change_naming_formatter(_=None):
    """Handle user update of naming formatter."""
    if CBB.blocking():
        return
    if not CM["input_naming"].validate():
        CM.update("input_naming", "FX.LOM{CH:02d}.Z")
    if _is_new():
        CM["overwrite_naming"] = {}
    _refresh_layout()


def _on_change_overwrite_naming(e):
    """Handle naming change from table."""
    row = e.args  # props.row
    ch, new_naming = row['channel'], row["naming"]
    default = CM["input_naming"].value.format(CH=ch)
    if new_naming == default:
        CM["overwrite_naming"].pop(ch, None)
    else:
        if is_valid_naming(new_naming):
            CM["overwrite_naming"][ch] = new_naming
        else:
            # Restore and refresh UI
            row["naming"] = default
            CM["table_layout"].rows = CM["table_layout"].rows


######
# IO #
######


def _is_new():
    """Return True if current selected session is '<NEW>'."""
    return CM["select_session"].value == "<NEW>"


def _check_save(_=None):
    """Trigger save validation."""
    name_exists = bool(CM["input_name"].value.strip())
    name_valid = CM["input_name"].validate()
    layout_valid = CM["is_layout_valid"]

    is_valid = name_exists and name_valid and layout_valid

    # Save button enable/disable
    CM.update("button_save", props="disable", props_remove=is_valid)

    # --- Warning text visibility & content ---
    if not is_valid:
        lines = []
        if not name_exists:
            lines.append("  • Session Name is required.")
        if not name_valid:
            lines.append("  • Session Name is invalid.")
        if not layout_valid:
            lines.append("  • Layout is invalid.")
        CM.update("text_warn", "\n" + "\n\n".join(lines),
                  classes="hidden", classes_remove=True)
    else:
        CM.update("text_warn", classes="hidden")


def _save_session(_=None):
    """Save current session and update selection."""
    name = CM["input_name"].value.strip()
    lics = CM["select_lics"].value
    folder = DATA_DIR / lics / name
    folder.mkdir(parents=True, exist_ok=True)
    json_path = folder / "ui_state.json"

    # --- Abort if file already exists ---
    if json_path.exists():
        ui.notify(f'Session "{name}" already exists. Save aborted.', color='negative')
        return

    # --- Construct data dictionary ---
    data = {
        "name": CM["input_name"].value,
        "layout": {
            "method": CM["select_layout"].value,
            "grid": {
                "count_x": int(CM["number_nx"].value),
                "count_y": int(CM["number_ny"].value),
                "origin_x": int(CM["number_ox"].value),
                "origin_y": int(CM["number_oy"].value),
                "delta_x": int(CM["number_dx"].value),
                "delta_y": int(CM["number_dy"].value),
                "shear_x": int(CM["number_sx"].value),
                "shear_y": int(CM["number_sy"].value),
            },
            "function": {
                "code": CM["code_custom"].value.strip() + "\n",
            },
            "shift": CM["shift_layout"],
            "naming": {
                "template": CM["input_naming"].value,
                "overwrite": CM["overwrite_naming"]
            }
        },
        "source_trailing": {
            "enabled": CM["checkbox_st"].value,
            "shift_x": int(CM["number_st_x"].value),
            "shift_y": int(CM["number_st_y"].value),
            "channel": int(CM["number_st_ch"].value),
            "naming": CM["input_st_naming"].value
        },
        "datalogger": {
            "device": _logger_value2name(CM["select_device"].value),
            "datatype": CM["select_datatype"].value,
            "samplerate": int(CM["number_sr"].value),
            "duration": int(CM["number_duration"].value)
        },
        "source": {
            "excitation": CM["select_excitation"].value,
            "direction": CM["select_direction"].value,
            "coupling": CM["select_coupling"].value,
            "repeats": int(CM["number_repeats"].value)
        },
        "conditions": {
            "weather": CM["select_weather"].value,
            "temperature": CM["select_temperature"].value,
            "moisture": CM["select_moisture"].value,
            "texture": CM["select_texture"].value,
            "order": CM["select_order"].value,
            "agriculture": CM["select_agriculture"].value,
            "crop": CM["select_crop"].value,
            "cultivation": CM["select_cultivation"].value,
        },
        "notes": CM["input_notes"].value.strip(),
        "create_time": datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    }

    # --- Write to JSON file ---
    try:
        with open(json_path, "w", encoding="utf-8") as fs:
            json.dump(data, fs, indent=2)  # noqa
        ui.notify(f'Session saved to {json_path}.', color='positive')
    except Exception as e:  # noqa
        ui.notify(f'Failed to save {json_path}: {e}', color='negative')
        return

    # --- Add new LICS to dropdown options ---
    options = CM["select_session"].options
    CM.update("select_session", name,
              options=["<NEW>"] + [name] + options[1:])


def _load_session(json_path, input_name):
    """Load session from json"""
    with open(json_path, "r") as fs:
        data = json.load(fs)

    with CBB.block():
        CM.update("input_name", input_name)

        # Layout
        CM.update("select_layout", data["layout"]["method"])
        CM.update("number_nx", data["layout"]["grid"]["count_x"])
        CM.update("number_ny", data["layout"]["grid"]["count_y"])
        CM.update("number_ox", data["layout"]["grid"]["origin_x"])
        CM.update("number_oy", data["layout"]["grid"]["origin_y"])
        CM.update("number_dx", data["layout"]["grid"]["delta_x"])
        CM.update("number_dy", data["layout"]["grid"]["delta_y"])
        CM.update("number_sx", data["layout"]["grid"]["shear_x"])
        CM.update("number_sy", data["layout"]["grid"]["shear_y"])
        CM.update("code_custom", data["layout"]["function"]["code"])
        # Ensure keys are int and values are (int, int) tuples
        CM["shift_layout"] = {
            int(ch): tuple(map(int, shift))
            for ch, shift in data["layout"]["shift"].items()
        }
        CM.update("input_naming", data["layout"]["naming"]["template"])
        CM["overwrite_naming"] = {
            int(ch): value
            for ch, value in data["layout"]["naming"]["overwrite"].items()
        }

        # Source Trailing
        CM.update("checkbox_st", data["source_trailing"]["enabled"])
        CM.update("number_st_x", data["source_trailing"]["shift_x"])
        CM.update("number_st_y", data["source_trailing"]["shift_y"])
        # We didn't save min, so cannot restore it.
        # Using 1 here and update layer by _on_change_layout_params().
        CM["number_st_ch"].min = 1
        CM.update("number_st_ch", data["source_trailing"]["channel"])
        CM.update("input_st_naming", data["source_trailing"]["naming"])

        # Datalogger
        CM.update("select_device", _logger_name2value(data["datalogger"]["device"]))
        CM.update("select_datatype", data["datalogger"]["datatype"])
        CM.update("number_sr", data["datalogger"]["samplerate"])
        CM.update("number_duration", data["datalogger"]["duration"])

        # Source
        CM.update("select_excitation", data["source"]["excitation"])
        CM.update("select_direction", data["source"]["direction"])
        CM.update("select_coupling", data["source"]["coupling"])
        CM.update("number_repeats", data["source"]["repeats"])

        # Conditions
        CM.update("select_weather", data["conditions"]["weather"])
        CM.update("select_temperature", data["conditions"]["temperature"])
        CM.update("select_moisture", data["conditions"]["moisture"])
        CM.update("select_texture", data["conditions"]["texture"])
        CM.update("select_order", data["conditions"]["order"])
        CM.update("select_agriculture", data["conditions"]["agriculture"])
        CM.update("select_crop", data["conditions"]["crop"])
        CM.update("select_cultivation", data["conditions"]["cultivation"])

        # Notes and time
        CM.update("input_notes", data["notes"])
        CM.update("input_time", data["create_time"])

    # Must manually refresh
    _on_change_layout_params()  # Will discard location shifting in New
    _on_change_naming_formatter()  # Will discard naming overwriting in New


def _restore_for_new():
    """Try restore from existing for new."""
    lics = CM["select_lics"].value
    try:
        # First, try last selection
        name = CM['last_selection']
        json_path = DATA_DIR / lics / name / "ui_state.json"
        _load_session(json_path, input_name="")
    except Exception as e:  # noqa
        try:
            # Second, try latest creation
            name = CM['select_session'].options[1]
            json_path = DATA_DIR / lics / name / "ui_state.json"
            _load_session(json_path, input_name="")
        except Exception as e:  # noqa
            # Finally, fall back to default
            _load_session("src/ui/defaults/default_session.json", input_name="")

    # For safety
    _check_save()


def _on_change_select_session(_=None):
    """Handle user changing selected session, including loading saved session data."""
    is_new = _is_new()

    # 1. Set all field readonly states based on selection
    for key in ["input_name",
                "select_layout",
                "number_nx", "number_ny", "number_ox", "number_oy",
                "number_dx", "number_dy", "number_sx", "number_sy",
                "select_device", "select_datatype", "number_sr", "number_duration",
                "select_excitation", "select_direction", "select_coupling", "number_repeats",
                "number_st_x", "number_st_y", "number_st_ch",
                "select_weather", "select_temperature",
                "select_moisture", "select_texture", "select_order",
                "select_agriculture", "select_crop", "select_cultivation",
                "input_notes"]:
        CM.update(key, props="readonly", props_remove=is_new)
    CM.update("code_custom", props="disable", props_remove=is_new)
    CM.update("checkbox_st", props="disable", props_remove=is_new)

    # 2. Toggle visibility of controls
    CM.update("checkbox_04d", classes="hidden", classes_remove=is_new)
    CM.update("input_time", classes="hidden", classes_remove=not is_new)
    CM.update("button_save", classes="hidden", classes_remove=is_new)
    CM.update("text_warn", classes="hidden", classes_remove=is_new)

    # 3. If loading existing session, populate fields from saved data
    lics = CM["select_lics"].value
    if not is_new:
        # Load selected
        name = CM["select_session"].value
        json_path = DATA_DIR / lics / name / "ui_state.json"
        try:
            _load_session(json_path, input_name=name)

            # Log the last valid selection
            CM["last_selection"] = name
        except Exception as e:  # noqa
            ui.notify(f'Failed to load/parse "{json_path}": {e}', color='negative')

            # Update options and selection
            session_options = ["<NEW>"] + get_existing_sorted(DATA_DIR / lics)
            CM.update("select_session", "<NEW>", options=session_options)
    else:
        # Load previous with fallback: last selection -> latest creation -> default
        _restore_for_new()


##############
# Input Name #
##############


NAME_VALIDATE_LEGACY = {
    "Name cannot be '<NEW>'": lambda name: name.strip() != "<NEW>",
    "Invalid session name format": lambda name: (
            not name.strip()
            or bool(re.fullmatch(r'session_\d{4}', name))
            or bool(re.fullmatch(r'0*[1-9]\d{0,3}', name))
    )
}

NAME_VALIDATE_FREE = {
    "Name cannot be '<NEW>'": lambda name: name.strip() != "<NEW>"
}


def _on_change_name_format(e):
    """Handle user update of name format."""
    if e.value:
        CM["input_name"].validation = NAME_VALIDATE_LEGACY
    else:
        CM["input_name"].validation = NAME_VALIDATE_FREE
    _check_save()


def _on_blur_name():
    """Autocorrect session name."""
    if CM["checkbox_04d"].value:
        name = CM["input_name"].value.strip()
        if re.fullmatch(r'0*[1-9]\d{0,3}', name):
            CM.update("input_name", f"session_{int(name):04d}")


###################
# Source Trailing #
###################


def _on_change_naming_formatter_st(_=None):
    """Handle user update of naming formatter of source trailing."""
    if not CM["input_st_naming"].validate():
        CM.update("input_st_naming", "ST.LOM{CH:02d}.Z")
    CM.update("input_st_naming_result",
              CM["input_st_naming"].value.format(CH=int(CM["number_st_ch"].value)))


def _on_change_st(e):
    """Source trailing checkbox disabled/enabled."""
    enabled = e.value
    for key in ["number_st_x", "number_st_y", "number_st_ch",
                "input_st_naming", "input_st_naming_result"]:
        CM.update(key, props="disable", props_remove=enabled)
    _check_channels()


def _on_change_st_channel(_=None):
    """Source trailing channel change."""
    _on_change_naming_formatter_st()
    _check_channels()


##################################
# Logger Validation & Monitoring #
##################################


def _check_channels():
    """Check if device provides enough channels."""
    device_name, n_ch_device = _get_selected_device_nch()
    chs_device = list(range(1, n_ch_device + 1))

    layout = _compute_layout()
    if layout is None:  # Code invalid
        return
    n_ch_layout = len(layout)
    chs_layout = list(range(1, n_ch_layout + 1))
    chs_required = chs_layout.copy()

    chs_st = None
    if CM["checkbox_st"].value:
        try:
            chs_st_val = int(CM["number_st_ch"].value)
            if chs_st_val > 0:
                chs_st = chs_st_val
                chs_required.append(chs_st_val)
        except:  # noqa
            pass

    lines = [
        f"• Channels Available on Device: {chs_device}",
        f"• Channels Requested by Layout: {chs_layout}",
        f"• Channels Requested by Trailing: {chs_st if chs_st else 'None'}"
    ]

    diff = set(chs_required) - set(chs_device)
    if diff:
        lines.append(f"• Channels Missing from Device: {sorted(diff)}")
        CM.update("expansion_check_ch", text="⚠️ Insufficient Channels on Device (but you are allowed to proceed)")
        CM.update("text_check_ch", value="\n  " + "\n\n  ".join(lines))
    else:
        lines.append("• All Channels Ready on Device.")
        CM.update("expansion_check_ch", text="🟢 Sufficient Channels on Device")
        CM.update("text_check_ch", value="\n  " + "\n\n  ".join(lines))


def _logger_name2value(name):
    """Format options in logger select."""
    n_ch = CM["detected_devices"][name]["n_chs"]
    if n_ch == 0:
        return f'{name}  【⚠️ Undetected】'
    else:
        return f'{name}  【🟢 {n_ch} CHs Ready】'


def _logger_value2name(value):
    """Extract name from select option."""
    return value.split("【")[0].strip()


def _get_selected_device_nch():
    """Get name and channels of selected device."""
    device_name = _logger_value2name(CM["select_device"].value)
    n_channels = CM["detected_devices"][device_name]["n_chs"]
    return device_name, n_channels


def _check_monitor(_=None):
    """Check if monitor is available."""
    device_name, n_channels = _get_selected_device_nch()
    CM.update("button_monitor", props="disable", props_remove=n_channels > 0)
    _check_channels()


def _refresh_device(_=None):
    """Refresh device table."""
    # Update device availability
    CM["detected_devices"] = Datalogger.get_logical_devices()

    # Value and options
    name_current = _logger_value2name(CM["select_device"].value)
    idx_current = None
    options = []
    for i, name in enumerate(CM["detected_devices"].keys()):
        options.append(_logger_name2value(name))
        if name == name_current:
            idx_current = i

    # Check index
    assert idx_current is not None and 0 <= idx_current < len(options), \
        "Impossible Error. Report for debugging."

    # Update
    CM.update("select_device", value=options[idx_current], options=options)
    CM["select_device"].update()

    # Check monitor
    _check_monitor()


def _monitor_device(_=None):
    """Open a dialog to monitor device signal in real time (time domain or frequency domain)."""

    # --- Step 1: Get current device settings from UI ---
    device_name, n_channels = _get_selected_device_nch()
    if n_channels == 0:
        ui.notify(f"Device '{device_name}' is not detected.", color='warning')
        return

    datatype = CM["select_datatype"].value
    samplerate = CM["number_sr"].value
    channel_list = list(range(1, n_channels + 1))

    # --- Step 2: Define UI and Datalogger instance ---
    datalogger = Datalogger()

    def _close_and_stop():
        """Stop stream and close dialog."""
        datalogger.stop_streaming()
        dlg.close()

    # Dialog and controls
    with ui.dialog().props('persistent') as dlg, ui.card().style('width: 80vw; height: 80vh;'):
        with ui.row().classes("items-center justify-between w-full"):
            ui.label(f"Monitoring: {device_name}").classes("text-2xl font-bold")
            ui.button(icon="close", on_click=_close_and_stop).classes("text-2xl").props("flat round").classes(
                "w-12 h-12")

        # Basic settings row
        with MyUI.row():
            refresh_interval = ui.number("Refresh Interval (s)", value=0.5, min=0.1, step=0.1).classes("flex-1")
            window_selector = ui.select(
                options=[256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
                value=1024,
                label="Window Size"
            ).classes("flex-1")

        # Frequency domain controls
        with MyUI.row():
            number_min_freq = ui.number("Min Freq (Hz)", value=100,
                                        min=0, max=samplerate // 2,
                                        step=100).classes("flex-1")
            number_max_freq = ui.number("Max Freq (Hz)", value=samplerate // 2,
                                        min=0, max=samplerate // 2,
                                        step=100).classes("flex-1")
            checkbox_freq = MyUI.checkbox("Frequency Domain", value=False, full=False)
            number_min_freq.bind_enabled_from(checkbox_freq, "value")
            number_max_freq.bind_enabled_from(checkbox_freq, "value")

        # Plot and control
        fig = ui.matplotlib(figsize=(6, 1.5 * max(n_channels / 2, 3))).classes("w-full self-center").figure

    dlg.open()

    # --- Step 3: Initialize buffer and callback ---
    signal_buffer = np.zeros((int(window_selector.value), n_channels), dtype=np.float32)
    last_draw = 0

    def _on_data(block: np.ndarray):
        nonlocal signal_buffer, last_draw

        # Update signal buffer
        window = int(window_selector.value)
        signal_buffer = np.vstack([signal_buffer, block])[-window:]
        if signal_buffer.shape[0] != window:
            return  # Wait until buffer is full

        # Control update frequency
        now = time.time()
        if now - last_draw < float(refresh_interval.value):
            return
        last_draw = now

        # Start plotting
        with fig:
            ax = fig.gca()
            ax.clear()

            if checkbox_freq.value:
                # --- Frequency domain ---
                freqs = np.fft.rfftfreq(window, d=1 / samplerate)
                fft_vals = np.abs(np.fft.rfft(signal_buffer, axis=0))

                try:
                    min_freq = float(number_min_freq.value or 0.0)
                    max_freq = float(number_max_freq.value or samplerate // 2)
                except:  # noqa
                    min_freq, max_freq = 0.0, samplerate // 2

                # Swap if invalid
                if min_freq > max_freq:
                    min_freq, max_freq = max_freq, min_freq

                idx = (freqs >= min_freq) & (freqs <= max_freq)
                freqs = freqs[idx]
                fft_vals = fft_vals[idx, :]
                if fft_vals.shape[0] == 0 or np.all(fft_vals == 0):
                    return  # Avoid invalid normalization or empty plot

                normed = fft_vals / (np.max(np.abs(fft_vals), axis=0, keepdims=True) + 1e-8) / 2
                for ch in range(n_channels):
                    ax.plot(freqs, -normed[:, ch] + ch, lw=1)
                ax.set_xlabel("Frequency (Hz)")

            else:
                # --- Time domain ---
                normed = signal_buffer / (np.max(np.abs(signal_buffer), axis=0, keepdims=True) + 1e-8) / 2
                time_axis = np.arange(window) / samplerate
                for ch in range(n_channels):
                    ax.plot(time_axis, -normed[:, ch] + ch, lw=1)
                ax.set_xlabel("Time (s)")

            ax.set_ylim(-1, n_channels)
            ax.set_yticks(np.arange(n_channels))
            ax.set_yticklabels([str(i + 1) for i in range(n_channels)])
            ax.invert_yaxis()
            fig.tight_layout(pad=0)
            MyPlot.apply_dark(fig)

    def _on_data_try(block: np.ndarray):
        try:
            _on_data(block)
        except Exception as e:
            print(f"[Monitor ERROR] {e}")

    # --- Step 4: Start streaming ---
    datalogger.start_monitoring(
        logical_name=device_name,
        channel_list=channel_list,
        datatype=datatype,
        samplerate=samplerate,
        on_data=_on_data_try
    )


##################
# FOR EXPERIMENT #
##################


def _get_trailing():
    """Return trailing information."""
    if CM["checkbox_st"].value:
        return {
            "shift_x": CM["number_st_x"].value,
            "shift_y": CM["number_st_y"].value,
            "channel": CM["number_st_ch"].value
        }
    else:
        return None


def _get_channel_naming():
    """Return a mapping from channel number to its naming."""
    ch_naming = {}
    for row in CM["table_layout"].rows:
        ch_naming[row["channel"]] = row["naming"]
    return ch_naming


def get_session_dict():
    """Return everything about session."""
    device_name, n_channels = _get_selected_device_nch()
    return {
        "layout": _compute_layout(),
        "st_dict": _get_trailing(),
        "naming": _get_channel_naming(),
        "device": {"name": device_name, "n_channels": n_channels}
    }


###########################
# MAIN UI INITIALIZATION  #
###########################


def _initialize_session_ui(e):
    """Render session tab based on selected lics."""
    lics = e.value
    CM["session_container"].clear()
    with CM["session_container"]:
        if lics == "<NEW>":
            ui.label('⚠️ Please select a LICS.').classes('text-xl')
            GS.selected_session = "<NEW>"
            return

        # --- Selection ---
        with MyUI.row():
            existing_sessions = get_existing_sorted(DATA_DIR / lics)
            sessions_options = ["<NEW>"] + existing_sessions

            # Selection
            CM["select_session"] = ui.select(
                sessions_options,
                value="<NEW>",
                on_change=_on_change_select_session,
                label="Select or Create a Session"
            ).classes('flex-1')

            # Global binding
            CM["select_session"].bind_value_to(GS, "selected_session")

            # LICS
            CM["select_lics"] = ui.input("Under LICS", value=lics). \
                props("readonly").classes('flex-1')

        # --- Name Input ---
        with MyUI.row():
            # Name
            CM["input_name"] = ui.input(
                "Name",
                on_change=_check_save,
                validation=NAME_VALIDATE_LEGACY,
            ).classes('flex-1').on("blur", _on_blur_name).props('autocomplete=off')

            # Name format
            CM["checkbox_04d"] = MyUI.checkbox(
                "Legacy naming: session_0001. Enter 1–9999.",
                value=True, on_change=_on_change_name_format, full=False)

        ###############
        # Layout Card #
        ###############
        with MyUI.cap_card("Layout"):
            with MyUI.row():
                with MyUI.row():
                    # Shift: Define at beginning to void null reference
                    CM["shift_layout"] = {}  # Data
                    CM["overwrite_naming"] = {}  # Data

                    # Method
                    CM["select_layout"] = ui.select(
                        ["GRID-1D/2D", "FUNCTION"],
                        label="Positioning Method", value="GRID-1D/2D",
                        on_change=_on_change_select_layout).classes('flex-1')

                    # Naming
                    CM["input_naming"] = ui.input(
                        label="Naming Formatter (Network.Station.Component)",
                        value="FX.LOM{CH:02d}.Z",
                        validation={"Invalid formatter": is_valid_naming}
                    ).classes('flex-1').on("blur", _on_change_naming_formatter)

                # --- GRID-1D/2D ---
                with MyUI.row() as CM["container_grid"]:
                    with ui.column().classes('flex-1'):
                        CM["number_nx"] = MyUI.number_int(
                            "Count X", min=1, value=7,
                            on_change=_on_change_layout_params)
                        CM["number_ny"] = MyUI.number_int(
                            "Count Y", min=1, value=1,
                            on_change=_on_change_layout_params)
                    with ui.column().classes('flex-1'):
                        CM["number_ox"] = MyUI.number_int(
                            "Origin X (cm)", value=0,
                            on_change=_on_change_layout_params)
                        CM["number_oy"] = MyUI.number_int(
                            "Origin Y (cm)", value=0,
                            on_change=_on_change_layout_params)
                    with ui.column().classes('flex-1'):
                        CM["number_dx"] = MyUI.number_int(
                            "Delta X (cm)", min=1, value=10,
                            on_change=_on_change_layout_params)
                        CM["number_dy"] = MyUI.number_int(
                            "Delta Y (cm)", min=1, value=0,
                            on_change=_on_change_layout_params)
                    with ui.column().classes('flex-1'):
                        CM["number_sx"] = MyUI.number_int(
                            "Shear X (cm)", value=0,
                            on_change=_on_change_layout_params)
                        CM["number_sy"] = MyUI.number_int(
                            "Shear Y (cm)", value=0,
                            on_change=_on_change_layout_params)

                # --- FUNCTION ---
                CM["code_custom"] = ui.codemirror(
                    value=DEFAULT_LAYOUT_CODE,
                    on_change=_on_change_layout_params,
                    language="Python",
                    theme="githubDark" if GS.dark_mode else "githubLight",
                ).classes("w-full hidden").style('height: 160px;')

            with MyUI.expansion(f"Layout Details: 【 7 Channels 】") as CM["expansion_layout"]:
                with MyUI.row(gap=4):
                    # --- Table and Figure ---
                    # Table
                    CM["table_layout"] = ui.table(
                        columns=[
                            {'name': 'channel', 'label': 'Channel', 'field': 'channel', 'align': 'left'},
                            {'name': 'x', 'label': 'X (cm)', 'field': 'x', 'align': 'left'},
                            {'name': 'y', 'label': 'Y (cm)', 'field': 'y', 'align': 'left'},
                            {'name': 'dx', 'label': 'Shift X (cm)', 'field': 'dx', 'align': 'left'},
                            {'name': 'dy', 'label': 'Shift Y (cm)', 'field': 'dy', 'align': 'left'},
                            {'name': 'naming', 'label': 'Naming', 'field': 'naming', 'align': 'left'},
                        ],
                        rows=[],
                        row_key='channel',
                        pagination=8,
                    ).classes('flex-[4] q-table--col-auto-width')

                    # Callback of numbers in table
                    CM["table_layout"].add_slot('body-cell-dx', r'''
                        <q-td key="dx" :props="props">
                            <q-input
                                dense
                                type="number"
                                class="w-full max-w-[80px]"
                                v-model.number="props.row.dx"
                                :readonly="props.row.lock_shift"
                                @update:model-value="() => $parent.$emit('_on_change_shift', props.row)"
                                @blur="() => $parent.$emit('_on_blur_shift', props.row)"
                            />
                        </q-td>
                    ''')

                    CM["table_layout"].add_slot('body-cell-dy', r'''
                        <q-td key="dy" :props="props">
                            <q-input
                                dense
                                type="number"
                                class="w-full max-w-[80px]"
                                v-model.number="props.row.dy"
                                :readonly="props.row.lock_shift"
                                @update:model-value="() => $parent.$emit('_on_change_shift', props.row)"
                                @blur="() => $parent.$emit('_on_blur_shift', props.row)"
                            />
                        </q-td>
                    ''')
                    CM["table_layout"].on('_on_change_shift', _on_change_shift)
                    CM["table_layout"].on('_on_blur_shift', _on_blur_shift)

                    CM["table_layout"].add_slot('body-cell-naming', r'''
                        <q-td key="naming" :props="props">
                            <q-input
                                dense
                                class="w-full max-w-[120px]"
                                v-model="props.row.naming"
                                :readonly="props.row.lock_shift"
                                :rules="[
                                  val => !!val && /^\S+\.\S+\.\S+$/.test(val) || 'Invalid. Expecting *.*.*'
                                ]"
                                hide-bottom-space
                                @blur="() => $parent.$emit('_on_change_overwrite_naming', props.row)"
                            />
                        </q-td>
                    ''')
                    CM["table_layout"].on('_on_change_overwrite_naming', _on_change_overwrite_naming)

                    # Figure
                    CM["figure_layout"] = ui.matplotlib(dpi=200, figsize=(4, 4)).classes("flex-[3]").figure

        with MyUI.row():
            ###################
            # Source Trailing #
            ###################
            height_px = 310
            with MyUI.cap_card("Source Trailing", full=False, height_px=height_px):
                with MyUI.row():
                    CM["checkbox_st"] = MyUI.checkbox(
                        "Conduct Source Training", value=True, on_change=_on_change_st)
                    ui.input().style('visibility: hidden')  # Dummy for alignment
                with MyUI.row():
                    CM["number_st_x"] = MyUI.number_int("Shift X (cm)", value=0, full=False)
                    CM["number_st_y"] = MyUI.number_int("Shift Y (cm)", value=10, full=False)
                CM["number_st_ch"] = MyUI.number_int("Channel", min=8, value=8,
                                                     on_change=_on_change_st_channel)
                with MyUI.row():
                    CM["input_st_naming"] = ui.input(
                        label="Naming Formatter",
                        value="ST.LOM{CH:02d}.Z",
                        validation={"Invalid formatter": is_valid_naming}
                    ).classes('flex-1').on("blur", _on_change_naming_formatter_st)
                    CM["input_st_naming_result"] = ui.input(
                        label="Naming Result",
                        value="ST.LOM?.Z"
                    ).classes('flex-1').props("readonly")
                    _on_change_naming_formatter_st()

            ##############
            # Datalogger #
            ##############
            with MyUI.cap_card("Datalogger", full=False, height_px=height_px):
                # --- Datalogger ---
                CM["detected_devices"] = {}
                with MyUI.row(gap=4):
                    # No callback changing device
                    CM["select_device"] = ui.select(
                        ["Dummy"], value="Dummy", label="Device",
                        on_change=_check_monitor
                    ).classes('flex-1')

                    # Pop up devices and select Dummy
                    _refresh_device()

                    # Let user refresh device detection
                    with ui.row().style('align-items: center; height: 56px;'):
                        ui.button(
                            icon="refresh",
                            on_click=_refresh_device
                        ).classes('w-8 h-8')

                    # Let user monitor selected device
                    with ui.row().style('align-items: center; height: 56px;'):
                        CM["button_monitor"] = ui.button(
                            icon="monitor_heart",
                            on_click=_monitor_device
                        ).classes('w-8 h-8')

                CM["select_datatype"] = ui.select(
                    SESSION_OPTIONS["datatype"], value="float32",
                    label="Datatype (float32 Recommended)").classes('w-full')
                CM["number_sr"] = MyUI.number_int("Sampling Rate", min=1, value=10000)
                CM["number_duration"] = MyUI.number_int("Duration (s)", min=1, value=5)

            ##########
            # Source #
            ##########
            with MyUI.cap_card("Source", full=False, height_px=height_px):
                CM["select_excitation"] = ui.select(
                    SESSION_OPTIONS["excitation"], value=SESSION_OPTIONS["excitation"][0],
                    label="Excitation").classes('w-full')
                CM["select_direction"] = ui.select(
                    SESSION_OPTIONS["direction"], value=SESSION_OPTIONS["direction"][-1],
                    label="Direction").classes('w-full')
                CM["select_coupling"] = ui.select(
                    SESSION_OPTIONS["coupling"], value=SESSION_OPTIONS["coupling"][0],
                    label="Coupling").classes('w-full')
                CM["number_repeats"] = MyUI.number_int("Repeats", min=1, value=5)

        ###############
        # Option Card #
        ###############
        def _create_static_options(key, show_text):
            options = ["Unknown"] + SESSION_OPTIONS.get(key, [])
            CM[f"select_{key}"] = ui.select(
                options,
                label=show_text,
                value="Unknown"
            ).classes('flex-1')

        with MyUI.cap_card("Conditions"):
            with MyUI.row():
                _create_static_options("weather", "Weather Condition")
                _create_static_options("temperature", "Air Temperature")
                _create_static_options("moisture", "Soil Moisture")
                _create_static_options("texture", "Soil Texture")

            with MyUI.row():
                _create_static_options("order", "Soil Order")
                _create_static_options("agriculture", "Agricultural System")
                _create_static_options("crop", "Crop Type")
                _create_static_options("cultivation", "Cultivation Method")

        # --- Notes Input ---
        CM["input_notes"] = ui.input("Notes").classes('w-full')

        # --- Create Time ---
        CM["input_time"] = ui.input("Create Time").props("readonly").classes('w-full hidden')

        with ui.row().classes('w-full gap-10'):
            # --- Save Button ---
            CM["button_save"] = ui.button(
                "SAVE SESSION", color="primary", on_click=_save_session) \
                .props("disable").classes('w-1/4 text-white font-semibold h-16')

            # --- Warning Box ---
            CM["text_warn"] = ui.textarea(
                label="⚠️ Saving disabled due to:",
                value="\n  • Session Name is required."
            ).props('readonly borderless') \
                .classes('w-1/4 text-base large-label h-16')

            # --- Channel Check ---
            with MyUI.expansion("⚠️ Insufficient Channels on Device") as CM["expansion_check_ch"]:
                CM["text_check_ch"] = ui.textarea(
                    label="Channel Info"
                ).props('readonly borderless rows=10') \
                    .classes('w-full text-base large-label')

    # Load previous with fallback: last selection -> latest creation -> default
    _restore_for_new()


def initialize():
    """Initialize the session tab UI."""
    # ui.label("Define SESSION").classes('text-3xl font-bold')

    # Define container where dynamic UI will render
    CM["session_container"] = ui.column().classes('w-full')

    # Bind selected_lics and trigger render
    ui.input(on_change=_initialize_session_ui).classes('hidden') \
        .bind_value_from(GS, "selected_lics")

    # Manually trigger UI rendering based on current selected_lics
    _initialize_session_ui(SimpleNamespace(value=GS.selected_lics))
