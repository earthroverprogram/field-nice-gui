import asyncio
import json
import os
import re
import subprocess
import time
import warnings
from datetime import datetime
from types import SimpleNamespace

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from nicegui import ui
from obspy import Stream, Trace, UTCDateTime

from src.device.datalogger import Datalogger
from src.ui import GS, DATA_DIR
from src.ui.session import get_session_dict
from src.ui.utils import ControlManager, MyPlot, MyUI, CallbackBlocker, ThreeImageViewer, _detect_snuffler

# from src.osc_lib.evo16 import set_preamp_gain_evo16

# --- UI Control Registry ---
CM = ControlManager()

# --- UI Callback Blocker (Batch Assigner) ---
CBB = CallbackBlocker()

############
# Defaults #
############


# Gain templates
with open('src/ui/defaults/gain_templates.yaml', 'r', encoding='utf-8') as f:
    GAIN_TEMPLATES = yaml.safe_load(f)


#############
# Auto Gain #
#############


def _compute_gain(distances):
    """Compute gain from template and `param`."""
    front_code = CM["code_gain_param"].value  # User can change this
    back_code = GAIN_TEMPLATES[CM["select_gain"].value]["back_code"]
    validate_code = GAIN_TEMPLATES[CM["select_gain"].value]["validate_code"]

    try:
        local_env = {}

        # --- Execute user-defined parameter definition (e.g. param = ...) ---
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            exec(front_code, {}, local_env)
        if "param" not in local_env:
            raise ValueError("You must define a variable named 'param'.")
        param = local_env["param"]

        # --- Run validation function and capture error message if any ---
        exec(validate_code, {}, local_env)
        if "validate_param" not in local_env:
            raise ValueError("Validation function 'validate_param(param)' is not defined.")
        validate_error = local_env["validate_param"](param)
        if validate_error:
            raise ValueError(validate_error)

        # --- Compile compute_gain_factory and create gain function ---
        exec(back_code, {}, local_env)
        if "compute_gain_factory" not in local_env:
            raise ValueError("Function 'compute_gain_factory(param)' is not defined.")

        # --- Run the gain function on the distances ---
        compute_gain = local_env["compute_gain_factory"](param)
        gains = compute_gain(distances)

        if not isinstance(gains, list) or len(gains) != len(distances):
            raise ValueError("compute_gain() must return `list` "
                             "with same length as distances.")

    except Exception as e:  # noqa
        gains = None
        error = f"Gain function error: {e}"
        # Only notify here
        ui.notify(error, color="negative")

    if gains is not None:
        gains = [round(gain, ndigits=1) if gain is not None else None
                 for gain in gains]
    return gains


def _plot_gain_curve(fig):
    """Plot gain curve."""
    if CBB.blocking():
        return
    fig.clear()

    # --- Try computing gain values from user-defined param ---
    distances = np.linspace(10, 300, 600)
    gains = _compute_gain(distances)

    # --- If failed, show error ---
    if gains is None:
        MyPlot.error(fig, "Error Gain")
        CM["is_gain_valid"] = False
    else:
        # --- Draw gain curve ---
        ax = fig.gca()
        ax.grid(True, color=MyUI.gray_color(), linewidth=0.5)
        ax.plot(distances, gains, color=MyUI.primary_color(), zorder=10, lw=2)
        ax.set_xlabel("Distance (cm)")
        ax.set_ylabel("Gain (dB)")
        fig.tight_layout(pad=0)
        MyPlot.apply_dark(fig)
        CM["is_gain_valid"] = True

    # check save
    _check_save()


def _compute_summary():
    """Compute summary of final experiment layout."""

    # --- Compute layout ---
    layout = CM["session_dict"]["layout"]
    if layout is None:
        ui.notify("Impossible Error. Report for debugging.", color="negative")
        return None

    # --- Get source location from UI ---
    src_xy = np.array([CM["number_x"].value, CM["number_y"].value], dtype=int)

    # --- Compute trailing (if any) ---
    st_dict = CM["session_dict"]["st_dict"]
    layout_with_st = layout
    if st_dict:
        shift_xy = np.array([st_dict["shift_x"], st_dict["shift_y"]], dtype=int)
        st_xy = src_xy + shift_xy
        st_dict["x"], st_dict["y"] = int(st_xy[0]), int(st_xy[1])
        layout_with_st = np.concatenate([layout, st_xy[None, :]], axis=0)

    # --- Compute gain distances ---
    distances = np.linalg.norm(layout_with_st - src_xy[None, :], axis=1)

    # Warn if any channel coincides with the source
    loc_zeros = np.where(np.isclose(distances, 0.0))[0]
    if loc_zeros.size > 0:
        channels = []
        for loc in loc_zeros:
            channel = loc
            if st_dict and loc == len(distances) - 1:
                channel = st_dict["channel"]
            channels.append(str(channel))
        warn = "The following channels coincide with the source: [" + ", ".join(channels) + "]"
        ui.notify(warn, color="warning")

    # Warn if any normal channel coincides with source trailing
    if st_dict:
        st_xy = np.array([st_dict["x"], st_dict["y"]])
        distances_st = np.linalg.norm(layout_with_st[:-1] - st_xy[None, :], axis=1)

        loc_st_zeros = np.where(np.isclose(distances_st, 0.0))[0]
        if loc_st_zeros.size > 0:
            channels = [str(idx + 1) for idx in loc_st_zeros]
            warn = f"The following channels coincide with source trailing: [{', '.join(channels)}]"
            ui.notify(warn, color="warning")

    # --- Compute gains ---
    gains = _compute_gain(distances)
    if gains is None:
        return None

    # --- Overwrite gains if manually set ---
    if CM["overwrite_gain"]:
        channels = list(CM["session_dict"]["naming"].keys())
        ch2idx = {ch: idx for idx, ch in enumerate(channels)}
        for ch, gain in CM["overwrite_gain"].items():
            if ch in ch2idx and ch2idx[ch] < len(gains):
                gains[ch2idx[ch]] = gain

    # --- Handle trailing gain (move last back to st_dict) ---
    if st_dict:
        st_dict["gain"] = gains[-1]
        gains = gains[:-1]  # Remove from layout

    # --- Return summary dict ---
    layout_dict = {
        "layout": layout,
        "gains": gains,
        "src_xy": src_xy,
        "st_dict": st_dict
    }
    return layout_dict


def _plot_summary(fig, layout_dict):
    """Plot final experiment layout with bypassed gains in no color."""
    fig.clear()

    if layout_dict is None:
        MyPlot.error(fig, "Error Gain")
        return

    layout = layout_dict["layout"]
    gains = layout_dict["gains"]
    src_xy = layout_dict["src_xy"]
    st_dict = layout_dict["st_dict"]
    bypass = [gain is None for gain in gains]

    # --- Valid gains for color normalization ---
    valid_gains = [g for g, b in zip(gains, bypass) if not b]
    if st_dict and st_dict["gain"] is not None:
        valid_gains.append(st_dict["gain"])
    if not valid_gains:
        valid_gains = [0]

    cmap = plt.get_cmap("rainbow")
    norm = Normalize(vmin=min(valid_gains), vmax=max(valid_gains))

    # --- Color assignment ---
    bypass_color = MyUI.font_color()
    colors = [
        bypass_color if is_bypass else cmap(norm(g))
        for g, is_bypass in zip(gains, bypass)
    ]

    # --- Plotting ---
    ax = fig.gca()
    ax.grid(True, color=MyUI.gray_color(), linewidth=0.5)
    ax.scatter(layout[:, 0], layout[:, 1], s=40, color=colors, zorder=10)
    ax.scatter(src_xy[0], src_xy[1], s=80, color='red', marker="*", zorder=20)
    if st_dict:
        gain = st_dict["gain"]
        color = bypass_color if gain is None else cmap(norm(gain))
        ax.scatter(st_dict["x"], st_dict["y"], s=30, color=color, marker="D", zorder=10)

    # --- Colorbar on top ---
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, orientation="horizontal",
                        location="top", pad=0.05)
    cbar.set_label("Gain (dB)  ", ha="right", x=0, labelpad=-25)

    # --- Margin ---
    points = [layout, src_xy[None, :]]
    if st_dict:
        points.append(np.array([[st_dict["x"], st_dict["y"]]]))
    all_xy = np.vstack(points)
    MyPlot.apply_margin(fig, all_xy)
    fig.tight_layout(pad=0)
    bbox = ax.get_position()
    ax.set_position([
        bbox.x0 + 0.1,
        bbox.y0,
        bbox.width,
        bbox.height
    ])
    bbox = cbar.ax.get_position()
    cbar.ax.set_position([
        bbox.x0 + 0.1,
        bbox.y0,
        bbox.width,
        bbox.height
    ])
    MyPlot.apply_dark(fig)


def _refresh_summary():
    """Update summary from scratch."""
    if CBB.blocking():
        return
    summary_dict = _compute_summary()
    if summary_dict is None:
        # Expansion
        CM["expansion_summary"].text = "⚠️ Error Gain"

        # Table
        CM["table_summary"].rows = []
        CM["table_summary"].update()
        CM["table_summary"].style(f"background-color: {MyUI.bg_color(True)}")
    else:
        def _parse_gain(gain_):
            """Gain => (Gain, Bypass)"""
            if gain_ is None:
                return -999, True
            return gain_, False

        # Expansion
        CM["expansion_summary"].text = \
            f"Summary Details: 【 {len(CM['session_dict']['naming'])} Channels 】"
        # Table
        new_rows = []
        for idx, ((x, y), gain) in enumerate(
                zip(summary_dict["layout"], summary_dict["gains"])):
            gain, bypass = _parse_gain(gain)
            new_rows.append({
                'channel': idx + 1,
                'x': x,
                'y': y,
                'gain': gain,
                'bypass': bypass,
                'lock_gain': not _is_new()
            })
        if summary_dict["st_dict"]:
            gain, bypass = _parse_gain(summary_dict["st_dict"]["gain"])
            new_rows.append({
                'channel': summary_dict['st_dict']['channel'],
                'x': summary_dict["st_dict"]["x"],
                'y': summary_dict["st_dict"]["y"],
                'gain': gain,
                'bypass': bypass,
                'lock_gain': not _is_new()
            })
        CM["table_summary"].clear()
        CM["table_summary"].rows = new_rows
        CM["table_summary"].update()
        CM["table_summary"].style("background-color: var(--q-surface)")

    # Figure
    with CM["figure_summary"]:
        _plot_summary(CM["figure_summary"], summary_dict)
    with CM["figure_summary_dup"]:
        _plot_summary(CM["figure_summary_dup"], summary_dict)


def _on_change_source_location(_=None):
    """Handle source location change."""
    _refresh_summary()
    _check_save()


def _on_change_gain_select(e):
    """Handle gain template change."""
    comment = "# Only provide `param` here\n"
    CM.update("code_gain_param", comment + GAIN_TEMPLATES[e.value]["front_code"])
    CM.update("text_gain_note", "ⓘ   " + GAIN_TEMPLATES[e.value]["note"])


def _on_change_gain_code(_=None):
    """Handle gain code change."""
    with CM["figure_gain"]:
        _plot_gain_curve(CM["figure_gain"])
    _refresh_summary()


def _on_change_summary_value(e):
    """Handle gain value change from table."""
    row = e.args  # props.row
    ch, gain = row['channel'], row["gain"]
    try:
        gain = round(float(gain), 1)
    except:  # noqa
        return  # Do nothing, wait user to finish
    CM["overwrite_gain"][ch] = gain
    _refresh_summary()


def _on_blur_summary_value(e):
    """Handle gain value blur from table."""
    row = e.args  # props.row
    ch, gain = row['channel'], row["gain"]
    try:
        _ = round(float(gain), 1)
        # Do nothing, changes already honored by _on_change_summary_value
    except:  # noqa
        CM["overwrite_gain"].pop(ch, None)
        _refresh_summary()


def _on_change_summary_bypass(e):
    """Handle gain bypass change from table."""
    row = e.args  # props.row
    ch, bypass = row['channel'], row["bypass"]
    if bypass:
        # Do bypass: set gain to None
        CM["overwrite_gain"][ch] = None
    else:
        # Undo bypass
        if CM["select_gain"].value == "Bypass":
            # Must give it a value or deadlock
            CM["overwrite_gain"][ch] = 0.0
        else:
            # Remove from overwrite and let function compute
            CM["overwrite_gain"].pop(ch, None)
    _refresh_summary()


######
# IO #
######

def _get_existing_experiments_sorted(session_path):
    """Return list of existing experiment folder *numbers* sorted descending (int)."""
    experiments = []
    pattern = re.compile(r"experiment_(\d{4})$")
    for d in session_path.iterdir():
        if d.is_dir():
            match = pattern.match(d.name)
            if not match:
                continue
            try:
                json_path = d / "ui_state.json"
                with open(json_path, "r", encoding="utf-8") as fs:
                    _ = json.load(fs)
                number = int(match.group(1))
                experiments.append(number)
            except Exception:  # noqa
                continue  # skip unreadable or malformed entries
    return sorted(experiments, reverse=True)


def _is_new():
    """Return True if current selected experiment is not in session folder."""
    number, folder = _get_experiment_folder()
    return (CM["number_experiment"].value not in
            _get_existing_experiments_sorted(folder.parent))


def _check_save(_=None):
    """Trigger save validation."""
    if not _is_new():
        CM.update("button_record", props="disable")
        CM.update("label_final", text="✅ Completed Experiment")
        loc_string = (f'#{int(CM["number_experiment"].value):04d} '
                      f'@ ({int(CM["number_x"].value)}, {int(CM["number_y"].value)})')
        CM.update("text_final", text=loc_string)
        return

    n_device_channels = CM["session_dict"]["datalogger"]["n_channels"]
    device_channels = set(range(1, n_device_channels + 1))
    request_channels = set(CM["session_dict"]["naming"].keys())
    diff = set(request_channels) - set(device_channels)

    is_channel_enough = len(diff) == 0
    if CM["checkbox_check_ch"]:
        is_channel_enough = is_channel_enough or not CM["checkbox_check_ch"].value
    is_valid = CM["is_gain_valid"] and is_channel_enough

    # --- Save button enable/disable ---
    CM.update("button_record", props="disable", props_remove=is_valid)

    # --- Warning text visibility & content ---
    if not is_valid:
        CM.update("label_final", text="⚠️ Recording Disabled")
        reasons = []
        if not CM["is_gain_valid"]:
            reasons.append("Invalid Gain")
        if not is_channel_enough:
            reasons.append("Insufficient Channels")
        CM.update("text_final", text="; ".join(reasons))
    else:
        CM.update("label_final", text="🟢 Ready to Record")
        loc_string = (f'#{int(CM["number_experiment"].value):04d} '
                      f'@ ({int(CM["number_x"].value)}, {int(CM["number_y"].value)})')
        CM.update("text_final", text=loc_string)


def _number_to_dir(experiment_number):
    """Path format."""
    return f"experiment_{int(experiment_number):04d}"


def _get_experiment_folder():
    """Get experiment folder."""
    lics = CM["select_lics"].value
    session = CM["select_session"].value
    number = int(CM["number_experiment"].value)
    folder = DATA_DIR / lics / session / _number_to_dir(number)
    return number, folder


def _save_experiment():
    """Save current experiment."""
    number, folder = _get_experiment_folder()
    folder.mkdir(parents=True, exist_ok=True)
    json_path = folder / "ui_state.json"

    # --- Abort if file already exists ---
    if json_path.exists():
        ui.notify(f'Experiment "{number}" already exists. Save aborted.', color='negative')
        return False

    # --- Construct data dictionary ---
    data = {
        "number": int(CM["number_experiment"].value),
        "source_location": {
            "x": int(CM["number_x"].value),
            "y": int(CM["number_y"].value)
        },
        "operators": {
            "computer": CM["input_computer_op"].value,
            "source": CM["input_source_op"].value,
            "protocol": CM["input_protocol_op"].value,
            "others": CM["input_others_op"].value
        },
        "gain": {
            "template": CM["select_gain"].value,
            "code_param": CM["code_gain_param"].value,
            "overwrite": {
                k: float(f"{v:.1f}") if v is not None else None
                for k, v in CM["overwrite_gain"].items()
            }
        },
        "pre_notes": CM["input_pre_notes"].value.strip(),
        "post_notes": CM["input_post_notes"].value.strip(),
        "create_time": datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    }

    # --- Write to JSON file ---
    try:
        with open(json_path, "w", encoding="utf-8") as fs:
            json.dump(data, fs, indent=2)  # noqa
        ui.notify(f'Experiment meta saved to "{json_path}".', color='positive')
        return True
    except Exception as e:  # noqa
        ui.notify(f'Failed to save {json_path}: {e}', color='negative')
        return False


def _save_post_notes():
    """Update post notes."""
    number, folder = _get_experiment_folder()
    json_path = folder / "ui_state.json"
    try:
        # Load
        with open(json_path, "r") as fs:
            data = json.load(fs)

        # Update
        data["post_notes"] = CM["input_post_notes"].value.strip()

        # Save
        with open(json_path, "w") as fs:
            json.dump(data, fs, indent=2)  # noqa
    except Exception as e:  # noqa
        ui.notify(f'Failed to save {json_path}: {e}', color='negative')


def _load_experiment(json_path, experiment_number, post_notes=True):
    """Load experiment from json"""
    with open(json_path, "r") as fs:
        data = json.load(fs)

    with CBB.block():
        CM.update("number_experiment", experiment_number)

        # Source location
        CM.update("number_x", data["source_location"]["x"])
        CM.update("number_y", data["source_location"]["y"])

        # Operators
        CM.update("input_computer_op", data["operators"]["computer"])
        CM.update("input_source_op", data["operators"]["source"])
        CM.update("input_protocol_op", data["operators"]["protocol"])
        CM.update("input_others_op", data["operators"]["others"])

        # Gain
        CM.update("select_gain", data["gain"]["template"])
        CM.update("code_gain_param", data["gain"]["code_param"])
        CM["overwrite_gain"] = {
            int(ch): value
            for ch, value in data["gain"]["overwrite"].items()
        }

        # Notes and time
        CM.update("input_pre_notes", data["pre_notes"])
        if post_notes:  # Post notes likely not shared across experiments
            CM.update("input_post_notes", data["post_notes"])
        else:
            CM.update("input_post_notes", "")
        CM.update("input_time", data["create_time"])

    # Must manually refresh gain and summary
    _on_change_gain_code()


def _restore_for_new():
    """Try restore from existing for new."""
    number, folder = _get_experiment_folder()
    current_number = CM['number_experiment'].value
    session_folder = folder.parent
    try:
        # First, try previous
        json_path = session_folder / _number_to_dir(current_number - 1) / "ui_state.json"
        _load_experiment(json_path, experiment_number=current_number, post_notes=False)
        from_previous = True
    except Exception as e:  # noqa
        try:
            # First, try last
            json_path = session_folder / _number_to_dir(CM["last_selection"]) / "ui_state.json"
            _load_experiment(json_path, experiment_number=current_number, post_notes=False)
            from_previous = False
        except Exception as e:  # noqa
            # Finally, fall back to default
            _load_experiment("src/ui/defaults/default_experiment.json",
                             experiment_number=current_number, post_notes=False)
            from_previous = False

    # Manually refresh
    _refresh_summary()

    # For increment
    return from_previous


def _allocate_new_experiment_number(folder):
    """Return the next of max of existing."""
    existing = _get_existing_experiments_sorted(folder)
    return max(existing) + 1 if existing else 1


def _on_change_experiment_number(_=None):
    """Handle user changing selected experiment, including loading saved session data."""
    # User used keyboard to clear number
    if CM["number_experiment"].value is None:
        return

    # New flag
    is_new = _is_new()

    # 1. Set all field readonly states based on selection
    for key in ["number_x", "number_y",
                "input_computer_op", "input_source_op", "input_protocol_op", "input_others_op",
                "select_gain",
                "select_excitation", "select_direction", "select_coupling", "number_repeats",
                "input_pre_notes"]:
        CM.update(key, props="readonly", props_remove=is_new)
    CM.update("input_post_notes", props="readonly", props_remove=not is_new)
    CM.update("code_gain_param", props="disable", props_remove=is_new)

    # Previous button
    number = int(CM["number_experiment"].value)
    CM.update("button_prev", props="disable", props_remove=number > 1)

    # 2. Toggle visibility of controls
    CM.update("input_time", classes="hidden", classes_remove=not is_new)

    # 3. If loading existing experiment, populate fields from saved data
    _, folder = _get_experiment_folder()
    if not is_new:
        # Load selected
        json_path = folder / "ui_state.json"
        try:
            _load_experiment(json_path, experiment_number=number)
        except Exception as e:  # noqa
            ui.notify(f'Failed to load/parse "{json_path}": {e}', color='negative')
            CM.update("number_experiment", _allocate_new_experiment_number(folder.parent))

        # Log the last valid selection
        CM["last_selection"] = CM['number_experiment'].value
    else:
        # Load previous with fallback: previous number -> latest creation -> default
        from_previous = _restore_for_new()

        # Source increment
        if from_previous:
            CM["number_x"].value += CM["number_inc_x"].value
            CM["number_y"].value += CM["number_inc_y"].value

    # Check save
    _check_save()

    # 4. Preview
    fallbacks = ["src/ui/defaults/preview.png"] * 3
    if number == 1:
        fallbacks[0] = ""  # Number 0 is not allowed
    CM["previewer"].set_images(
        folder.parent / f"experiment_{number - 1:04d}/preview.png",
        folder.parent / f"experiment_{number:04d}/preview.png",
        folder.parent / f"experiment_{number + 1:04d}/preview.png",
        fallbacks=fallbacks
    )
    data_exist = (folder.parent / f"experiment_{number:04d}/data.mseed").exists()
    CM.update("button_snuffler", props="disable", props_remove=data_exist)


def _view_in_snuffler(_=None):
    """Launch the Snuffler tool to view the corresponding .mseed file."""

    # Get the currently selected experiment number from the local state
    number, folder = _get_experiment_folder()
    mseed_path = folder.parent / f"experiment_{number:04d}/data.mseed"

    # Check if the .mseed file exists
    if not mseed_path.exists():
        ui.notify(f'{mseed_path} does not exist', type='negative')
        return

    try:
        # Launch Snuffler in a separate process (not blocking UI)
        subprocess.Popen([CM["input_path_snuffler"].value, str(mseed_path)])
    except Exception as e:
        ui.notify(f'Failed to launch Snuffler: {e}', type='negative')
    else:
        ui.notify(f'Snuffler launched for {mseed_path.name}', type='positive')


##########
# Record #
##########


def _save_data(data: np.ndarray):
    """
    Save multichannel recording data to experiment folder.
    - Format: MiniSEED
    - One trace per channel
    - Also generates waveform preview as PNG
    """

    # Step 1: Save experiment meta first
    if not _save_experiment():
        ui.notify("Failed to save experiment meta. Data saving also skipped.", color='negative')
        return

    # Step 2: Get target folder
    number, folder = _get_experiment_folder()

    # Step 3: Create obspy Stream and fill with traces
    stream = Stream()
    naming_dict = CM["session_dict"]["naming"]
    samplerate = CM["session_dict"]["datalogger"]["samplerate"]

    now = UTCDateTime.now()
    for ch_index, (logical_ch, naming) in enumerate(naming_dict.items()):
        if ch_index >= data.shape[1]:
            ui.notify(f"Channel mismatch: not enough data columns for {logical_ch}", color='negative')
            continue
        network, station, channel = naming.split(".")
        trace = Trace(data[:, ch_index].copy(), header={
            "sampling_rate": samplerate,
            "starttime": now,
            "network": network,
            "station": station,
            "channel": channel,
        })
        stream.append(trace)

    # Step 4: Save to MiniSEED
    try:
        stream.write(folder / "data.mseed", format="MSEED")
        ui.notify(f"Data saved to MiniSEED {folder / 'data.mseed'}.", color='positive')
    except Exception as e:
        ui.notify(f"Failed to write MiniSEED: {e}", color='negative')
        return

    # Step 5: Save preview plot
    try:
        plt.style.use('default')  # Preview figure is always white
        fig = plt.figure(dpi=200, figsize=(6.4, 4.8))
        ax = fig.gca()

        y_offset = 0
        offset_step = 1
        if len(stream) > 0:
            time_axis = list(np.arange(stream[0].data.shape[0]) / stream[0].stats.sampling_rate)
        else:
            ui.notify("Impossible Error. Report for debugging.", color="negative")
            return
        for trace in stream:
            data = trace.data.astype(float)
            normed = data / (np.max(np.abs(data)) or 1) * 0.5  # 0.5 to avoid overlap
            ax.plot(time_axis, -normed + y_offset, linewidth=0.5)  # -normed for inverted y-axis
            label = f'{trace.stats.network}.{trace.stats.station}.{trace.stats.channel} '
            ax.text(time_axis[0], y_offset, label, ha="right", va='center', fontsize=8)
            y_offset += offset_step

        ax.set_xlabel("Time (s)")
        ax.set_yticks([])
        ax.set_xlim(time_axis[0], time_axis[-1])
        lics = CM["select_lics"].value
        session = CM["select_session"].value
        number = int(CM["number_experiment"].value)
        ax.set_title(f"{lics}/{session}/{_number_to_dir(number)}")
        ax.invert_yaxis()  # Make ch=1 on top
        fig.tight_layout(pad=0)
        fig.savefig(folder / "preview.png", bbox_inches="tight")
        fig.clf()
        plt.close(fig)
        plt.style.use('dark_background' if GS.dark_mode else 'default')  # Restore plot theme
    except Exception as e:
        ui.notify(f"Failed to save preview image: {e}", color='negative')


async def _record():
    async def _reset_ui():
        CM["is_recording"] = False
        CM["button_record"].set_icon("radio_button_checked")
        CM["button_record"].props(remove="disable")
        await asyncio.sleep(0.2)
        CM["progress"].set_value(0.0)
        _on_change_experiment_number()

    def _start_record():
        CM["is_recording"] = True
        CM["button_record"].set_icon("stop_circle")
        CM["button_record"].props(remove="disable")
        CM["progress"].set_value(0.0)
        CM["datalogger"].start_recording(
            logical_name=logical_name,
            channel_list=channel_list,
            datatype=datatype,
            samplerate=samplerate,
        )

    # If currently recording: STOP
    if CM["is_recording"]:
        data = CM["datalogger"].stop_streaming()
        _save_data(data)
        await _reset_ui()
        return

    # Retrieve recording parameters
    info = CM["session_dict"]["datalogger"]
    logical_name = info["name"]
    duration = info["duration"]
    samplerate = info["samplerate"]
    datatype = info["datatype"]
    channel_list = list(CM["session_dict"]["naming"].keys())

    # set preamp on EVO-16
    if logical_name == "EVO-16":
        ch_gain = {}
        for row in CM["table_summary"].rows:
            ch_gain[row["channel"]] = row["gain"]
        try:
            Datalogger.set_preamp_gain(device=logical_name, channel_gain_dict=ch_gain)
        except Exception as ex:
            ui.notify(f"Failed to set preamp gain on EVO-16: {ex}", color='negative')
            await _reset_ui()
            return

    # Optional countdown before recording starts
    if CM["checkbox_countdown"].value:
        # Disable button during countdown
        CM["button_record"].props("disable")

        # Play voice prompt if enabled
        if CM['select_voice'].value != "<Silent>":
            CM["audio_countdown"].set_source(f"assets/countdown/{CM['select_voice'].value}.mp3")
            CM["audio_countdown"].play()
            await asyncio.sleep(0.1)  # ensure audio playback starts

        # Display countdown numbers: 3, 2, 1
        CM["display_countdown"].style("display: block;")
        for word in ["3", "2", "1"]:
            CM["display_countdown"].text = word
            await asyncio.sleep(1)

        # Display "Go!" and start recording immediately
        CM["display_countdown"].text = "Go!"
        try:
            _start_record()
        except Exception as e:  # recording failed
            ui.notify(f"Failed to start recording: {e}", color='negative')
            await _reset_ui()
            return
        await asyncio.sleep(1)
        CM["display_countdown"].style("display: none;")

        # Re-enable button after countdown
        CM["button_record"].props(remove="disable")

    else:
        # No countdown, start recording immediately
        try:
            _start_record()
        except Exception as e:  # recording failed
            ui.notify(f"Failed to start recording: {e}", color='negative')
            await _reset_ui()
            return

    # Start progress bar update loop
    start_time = time.perf_counter()
    while CM["is_recording"] and time.perf_counter() - start_time < duration:
        elapsed = time.perf_counter() - start_time
        CM["progress"].set_value(elapsed / duration)
        await asyncio.sleep(0.05)

    # Ensure final recording block is received before stopping
    await asyncio.sleep(0.5)

    # If not manually stopped, stop now (duration reached)
    if CM["is_recording"]:
        data = CM["datalogger"].stop_streaming()
        _save_data(data)
        await _reset_ui()

    # Go next
    if CM["checkbox_next"].value:
        await asyncio.sleep(0.1)  # Wait for figure to flush
        _go_next()


###################
# Local Interplay #
###################


def _go_previous():
    """Go previous experiment."""
    if CM["number_experiment"].value > 1:
        CM["number_experiment"].value -= 1


def _go_next():
    """Go next experiment."""
    CM["number_experiment"].value += 1


def _on_change_edit_snuffler(_=None):
    """User edit snuffler path."""
    if "snuffler" not in CM["input_path_snuffler"].value:
        CM["input_path_snuffler"].value = _detect_snuffler()


###########################
# MAIN UI INITIALIZATION  #
###########################


def _initialize_experiment_ui(_=None):
    """Render experiment tab based on selected lics and session."""
    lics = GS.selected_lics
    session = GS.selected_session
    CM["experiment_container"].clear()
    with (CM["experiment_container"]):
        if session == "<NEW>":
            ui.label('⚠️ Please select a Session.').classes('text-xl')
            return

        # Get everything of session
        CM["session_dict"] = get_session_dict()

        with MyUI.row():
            with ui.column().classes("flex-1"):
                # --- Experiment Number ---
                session_folder = DATA_DIR / lics / session
                number_init = _allocate_new_experiment_number(session_folder)
                CM["number_experiment"] = MyUI.number_int(
                    "Experiment Number",
                    min=1, value=number_init,
                    on_change=_on_change_experiment_number,
                )
            with ui.column().classes("flex-1"):
                with MyUI.row():
                    CM["select_session"] = ui.input("Under Session", value=session). \
                        props("readonly").classes('flex-1')
                    CM["select_lics"] = ui.input("Under LICS", value=lics). \
                        props("readonly").classes('flex-1')

        with MyUI.row():
            ###################
            # Source Location #
            ###################
            with MyUI.cap_card("Source Location", full=False, highlight=True):
                with MyUI.row():
                    CM["number_x"] = MyUI.number_int("Location X (cm)", value=-20,
                                                     on_change=_on_change_source_location, full=False)
                    CM["number_y"] = MyUI.number_int("Location Y (cm)", value=0,
                                                     on_change=_on_change_source_location, full=False)

            #############
            # Operators #
            #############
            with MyUI.cap_card("Operators", full=False):
                with MyUI.row():
                    CM["input_computer_op"] = ui.input("Computer").classes('flex-1')
                    CM["input_source_op"] = ui.input("Source").classes('flex-1')
                    CM["input_protocol_op"] = ui.input("Protocol").classes('flex-1')
                    CM["input_others_op"] = ui.input("We are just here").classes('flex-1')

        #############
        # Auto Gain #
        #############
        with MyUI.cap_card("Auto Gain"):
            with MyUI.expansion(f"Gain Function"):
                with MyUI.row():
                    with ui.column().classes('flex-[10] gap-1'):
                        default_gain_select = "Geometric spreading (physics-based)"
                        CM["select_gain"] = ui.select(
                            list(GAIN_TEMPLATES.keys()), value=default_gain_select,
                            label="Template", on_change=_on_change_gain_select).classes('w-full')
                        note = GAIN_TEMPLATES[default_gain_select]["note"]
                        CM["text_gain_note"] = ui.textarea(value="ⓘ   " + note) \
                            .props('disable rows=3') \
                            .props(f'input-style="color: {MyUI.font_color()}; padding-top: 0px; '
                                   'line-height: 2.0; margin-bottom: 10px"') \
                            .classes('w-full')
                        comment = "# Only provide `param` here\n"
                        CM["code_gain_param"] = ui.codemirror(
                            value=comment + GAIN_TEMPLATES[default_gain_select]["front_code"],
                            on_change=_on_change_gain_code,
                            language="Python",
                            theme="gruvboxDark" if GS.dark_mode else "gruvboxLight",
                        ).classes("w-full").style('height: 200px;')
                    with ui.matplotlib(dpi=200, figsize=(4, 4)) \
                            .classes("flex-[6]").figure as CM["figure_gain"]:  # noqa
                        _plot_gain_curve(CM["figure_gain"])

        #################
        # Final summary #
        #################
        with MyUI.cap_card("Summary"):
            with MyUI.expansion(f"Experiment Details: 【 0 Channels 】") as CM["expansion_summary"]:
                with MyUI.row(gap=4):
                    # --- Table and Figure ---
                    # Table
                    CM["table_summary"] = ui.table(
                        columns=[
                            {'name': 'channel', 'label': 'Channel', 'field': 'channel', 'align': 'left'},
                            {'name': 'x', 'label': 'X (cm)', 'field': 'x', 'align': 'left'},
                            {'name': 'y', 'label': 'Y (cm)', 'field': 'y', 'align': 'left'},
                            {'name': 'gain', 'label': 'Gain (db)', 'field': 'gain', 'align': 'left'},
                            {'name': 'bypass', 'label': 'Bypass Gain', 'field': 'bypass', 'align': 'left'},
                        ],
                        rows=[],
                        row_key='channel',
                        pagination=8
                    ).classes('flex-[4] q-table--col-auto-width')

                    # Shift
                    CM["overwrite_gain"] = {}  # Data

                    # Callback of numbers in table
                    CM["table_summary"].add_slot('body-cell-gain', r'''
                        <q-td key="gain" :props="props">
                            <q-input
                                dense
                                type="number"
                                class="w-full max-w-[80px]"
                                v-model.number="props.row.gain"
                                :step="0.1"
                                :readonly="props.row.lock_gain || props.row.bypass"
                                @update:model-value="() => $parent.$emit('_on_change_summary_value', props.row)"
                                @blur="() => $parent.$emit('_on_blur_summary_value', props.row)"
                            />
                        </q-td>
                    ''')
                    CM["table_summary"].on('_on_change_summary_value', _on_change_summary_value)
                    CM["table_summary"].on('_on_blur_summary_value', _on_blur_summary_value)

                    CM["table_summary"].add_slot('body-cell-bypass', r'''
                        <q-td key="bypass" :props="props">
                            <q-checkbox
                                dense
                                size="sm"
                                class="w-full max-w-[80px]"
                                v-model="props.row.bypass"
                                :disable="props.row.lock_gain"
                                color="primary"
                                @update:model-value="() => $parent.$emit('_on_change_summary_bypass', props.row)"
                            />
                        </q-td>
                    ''')
                    CM["table_summary"].on('_on_change_summary_bypass', _on_change_summary_bypass)

                    # Figure
                    CM["figure_summary"] = ui.matplotlib(dpi=200, figsize=(4, 4)).classes("flex-[3]").figure

            # --- Notes ---
            with MyUI.row():
                CM["input_pre_notes"] = ui.input("Pre-notes").classes('flex-1')
                CM["input_post_notes"] = ui.input("Post-notes") \
                    .classes('flex-1').props("readonly") \
                    .on("blur", _save_post_notes)

        ###########
        # Actions #
        ###########
        # Do not toggle time because we want static page during experiments
        # --- Create Time ---
        # CM["input_time"] = ui.input("Create Time").props("readonly").classes('w-full hidden')

        ui.element('div').style(f'height: 4px; background-color: {MyUI.gray_color()}; '
                                'border-radius: 2px; width: 100%; margin: 20px 0;')

        with MyUI.row():
            card_height = 300
            # --- Options ---
            with ui.column().classes('flex-[3]'):
                with MyUI.cap_card("Record Options", full=True, height_px=card_height):
                    with MyUI.row():
                        # Countdown options
                        CM["checkbox_countdown"] = MyUI.checkbox("Countdown", value=True, full=False)
                        files = sorted([audio[:-4] for audio in os.listdir('assets/countdown')
                                        if audio.endswith('.mp3')]) + ["<Silent>"]
                        CM["select_voice"] = ui.select(
                            files, value=np.random.choice(files[:-1])) \
                            .props('filled').classes('q-pt-none').classes('flex-1') \
                            .bind_enabled_from(CM["checkbox_countdown"], 'value')

                        # Hidden UI for countdown sound and visual
                        CM["audio_countdown"] = ui.audio("", autoplay=False, controls=False)
                        CM["display_countdown"] = ui.label("321").classes(
                            "fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 "
                            "text-primary text-[200px] font-bold z-50 pointer-events-none "
                            "rounded-xl px-8 py-4 text-center border-8"
                        ).style(
                            f"display: none; background-color: {MyUI.bg_color()}; min-width: 400px;"
                            f"border: 8px solid {MyUI.primary_color()};"
                        )

                    # Check channels
                    CM["checkbox_check_ch"] = MyUI.checkbox("Ensure Enough Channels",
                                                            value=True, full=True, on_change=_check_save)

                    # Go next after record
                    CM["checkbox_next"] = MyUI.checkbox("Go Next after Recording", value=True, full=True)

                    # Source increment
                    with MyUI.row():
                        CM["number_inc_x"] = MyUI.number_int("Src Incr X (cm)", value=10, full=False)
                        CM["number_inc_y"] = MyUI.number_int("Src Incr Y (cm)", value=0, full=False)

            # --- Record ---
            CM["is_recording"] = False
            CM["datalogger"] = Datalogger()
            with ui.column().classes('flex-[6]'):
                with MyUI.cap_card("Record", full=True, highlight=True, height_px=card_height):
                    with ui.row().classes('w-full items-center justify-center gap-4 flex-nowrap'):
                        CM["button_prev"] = ui.button(icon='chevron_left', on_click=_go_previous) \
                            .classes('text-6xl').props("flat round")

                        with ui.circular_progress(value=0.0, show_value=False,
                                                  size='6.5rem', color="negative") \
                                .props('track-color="transparent"') as CM["progress"]:
                            CM["button_record"] = ui.button(
                                icon='radio_button_checked',
                                on_click=_record
                            ).classes('text-6xl').props('flat round')

                        CM["button_next"] = ui.button(icon='chevron_right', on_click=_go_next) \
                            .classes('text-6xl').props("flat round")

                    # --- Info ---
                    CM["label_final"] = ui.label().classes('font-bold -mt-2 text-xl mr-2 font-mono w-full text-center')
                    CM["text_final"] = ui.label().classes('text-xl mr-2 font-mono w-full text-center')

            # --- Figure ---
            with ui.column().classes('flex-[3]'):
                with MyUI.cap_card("Final Setup", full=True, height_px=card_height):
                    CM["figure_summary_dup"] = ui.matplotlib(dpi=200, figsize=(4, 4)).classes("h-full").figure

        # Preview
        with MyUI.expansion("Output Preview", value=True).classes("w-full"):
            CM["previewer"] = ThreeImageViewer()

            # Snuffler
            with MyUI.row().classes("items-center w-full"):
                ui.input().classes("flex-1").style('visibility: hidden')
                with ui.row().classes("w-1/4 justify-center"):
                    CM["button_snuffler"] = ui.button("View in Snuffler", on_click=_view_in_snuffler) \
                        .classes('text-white font-semibold h-14 w-full')
                CM["input_path_snuffler"] = ui.input("Full Path to Snuffler", value=_detect_snuffler()) \
                    .classes('flex-1').on("blur", _on_change_edit_snuffler)

    # Safe call
    _on_change_experiment_number()


def initialize():
    """Initialize the experiment tab UI."""
    # ui.label("Run EXPERIMENT").classes('text-3xl font-bold')

    # Define container where dynamic UI will render
    CM["experiment_container"] = ui.column().classes('w-full')

    # Bind selected_session and trigger render
    ui.input(on_change=_initialize_experiment_ui).classes('hidden') \
        .bind_value_from(GS, "selected_session")

    # Manually trigger UI rendering based on current selected_session
    _initialize_experiment_ui(SimpleNamespace(value=GS.selected_session))
