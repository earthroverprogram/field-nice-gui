import asyncio
import json
import re
from datetime import datetime
from types import SimpleNamespace

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import os
import numpy as np
import yaml
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from nicegui import ui

from src.ui import GS, DATA_DIR
from src.ui.session import compute_layout, get_trailing, get_channel_to_idx
from src.ui.utils import ControlManager, MyPlot, MyUI, CallbackBlocker, ThreeImageViewer

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
        return False

    # --- Draw gain curve ---
    ax = fig.gca()
    ax.grid(True)
    ax.plot(distances, gains, color=MyUI.primary_color(), zorder=10, lw=2)
    ax.set_xlabel("Distance (cm)")
    ax.set_ylabel("Gain (dB)")
    fig.tight_layout(pad=0)
    MyPlot.apply_dark(fig)
    return True


def _compute_summary():
    """Compute summary of final experiment layout."""

    # --- Compute layout ---
    layout = compute_layout()
    if layout is None:
        ui.notify("Impossible Error. Report for debugging.", color="negative")
        return None

    # --- Get source location from UI ---
    src_xy = np.array([CM["number_x"].value, CM["number_y"].value], dtype=int)

    # --- Compute trailing (if any) ---
    st_dict = get_trailing()
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

    # Warn if any normal channel coincides with the trailing source
    if st_dict:
        st_xy = np.array([st_dict["x"], st_dict["y"]])
        distances_st = np.linalg.norm(layout_with_st[:-1] - st_xy[None, :], axis=1)

        loc_st_zeros = np.where(np.isclose(distances_st, 0.0))[0]
        if loc_st_zeros.size > 0:
            channels = [str(idx + 1) for idx in loc_st_zeros]
            warn = f"The following channels coincide with the trailing source: [{', '.join(channels)}]"
            ui.notify(warn, color="warning")

    # --- Compute gains ---
    gains = _compute_gain(distances)
    if gains is None:
        return None

    # --- Overwrite gains if manually set ---
    if CM["overwrite_gain"]:
        ch2idx = get_channel_to_idx()
        for ch, gain in CM["overwrite_gain"].items():
            if ch in ch2idx and ch2idx[ch] < len(gains):
                gains[ch2idx[ch]] = gain

    # --- Handle trailing gain (move last back to st_dict) ---
    if st_dict:
        st_dict["gain"] = gains[-1]
        gains = gains[:-1]  # Remove from layout

    # --- Return summary dict ---
    return {
        "layout": layout,
        "gains": gains,
        "src_xy": src_xy,
        "st_dict": st_dict
    }


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
    ax.grid(True)
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
    cbar.set_label("Gain (dB)")

    # --- Margin ---
    points = [layout, src_xy[None, :]]
    if st_dict:
        points.append(np.array([[st_dict["x"], st_dict["y"]]]))
    all_xy = np.vstack(points)
    MyPlot.apply_margin(fig, all_xy)
    fig.tight_layout(pad=0)
    MyPlot.apply_dark(fig)


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
            f"Summary Details: 【 {len(get_channel_to_idx())} Channels 】"
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

    # Review
    CM.update("text_final",
              f'\n#{int(CM["number_experiment"].value):04d} '
              f'@ ({int(CM["number_x"].value)}, {int(CM["number_y"].value)})',
              label="🟢 Final Review:")


def _on_change_source_location(_=None):
    """Handle source location change."""
    _refresh_summary()


def _on_change_gain_select(e):
    """Handle gain template change."""
    comment = "# Only provide `param` here\n"
    CM.update("code_gain_param", comment + GAIN_TEMPLATES[e.value]["front_code"])
    CM.update("text_gain_note", "ⓘ   " + GAIN_TEMPLATES[e.value]["note"])


def _on_change_gain_code(_):
    """Handle gain code change."""
    with CM["figure_gain"]:
        CM["is_gain_valid"] = _plot_gain_curve(CM["figure_gain"])
    _refresh_summary()
    _check_save()


def _on_change_summary_value(e):
    """Handle gain value change from table."""
    row = e.args  # props.row
    ch, gain = row['channel'], round(row["gain"], 1)
    CM["overwrite_gain"][ch] = gain
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

def _check_save(_=None):
    """Trigger save validation."""
    if not _is_new():
        CM.update("button_record", props="disable")
        return

    is_valid = CM["is_gain_valid"]

    # --- Save button enable/disable ---
    CM.update("button_record", props="disable", props_remove=is_valid)

    # --- Warning text visibility & content ---
    if not is_valid:
        lines = ["  • Gain is invalid."]
        CM.update("text_final", "\n" + "\n\n".join(lines),
                  label="⚠️ Recording disabled due to:")


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
            "y": int(CM["number_y"].value),
            "increment_x": int(CM["number_inc_x"].value),
            "increment_y": int(CM["number_inc_y"].value)
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
        ui.notify(f'Experiment saved to "{json_path}".', color='positive')
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
        CM.update("number_inc_x", data["source_location"]["increment_x"])
        CM.update("number_inc_y", data["source_location"]["increment_y"])

        # Operators
        CM.update("input_computer_op", data["operators"]["computer"])
        CM.update("input_source_op", data["operators"]["source"])
        CM.update("input_protocol_op", data["operators"]["protocol"])
        CM.update("input_others_op", data["operators"]["others"])

        # Gain
        CM.update("select_gain", data["gain"]["template"])
        CM.update("code_gain_param", data["gain"]["code_param"])
        CM["overwrite_gain"] = data["gain"]["overwrite"]

        # Notes and time
        CM.update("input_pre_notes", data["pre_notes"])
        if post_notes:  # Post notes likely not shared across experiments
            CM.update("input_post_notes", data["post_notes"])
        else:
            CM.update("input_post_notes", "")
        CM.update("input_time", data["create_time"])

    # Must manually refresh
    _refresh_summary()


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

    # For safety
    _check_save()

    # For increment
    return from_previous


def _get_next_number(folder):
    existing = _get_existing_experiments_sorted(folder)
    return max(existing) + 1 if existing else 1


def _on_change_experiment_number(_=None):
    """Handle user changing selected experiment, including loading saved session data."""
    is_new = _is_new()

    # 1. Set all field readonly states based on selection
    for key in ["number_x", "number_y", "number_inc_x", "number_inc_y",
                "input_computer_op", "input_source_op", "input_protocol_op", "input_others_op",
                "select_gain",
                "select_excitation", "select_direction", "select_coupling", "number_repeats",
                "input_pre_notes"]:
        CM.update(key, props="readonly", props_remove=is_new)
    CM.update("input_post_notes", props="readonly", props_remove=not is_new)
    CM.update("code_gain_param", props="disable", props_remove=is_new)
    _check_save()  # Button and Warn

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
            CM.update("number_experiment", _get_next_number(folder.parent))

        # Log the last valid selection
        CM["last_selection"] = CM['number_experiment'].value
    else:
        # Load previous with fallback: previous number -> latest creation -> default
        from_previous = _restore_for_new()

        # Source increment
        if CM["checkbox_increment"].value and from_previous:
            CM["number_x"].value += CM["number_inc_x"].value
            CM["number_y"].value += CM["number_inc_y"].value

    # 4. Preview
    fallbacks = ["src/ui/defaults/placeholder.png"] * 3
    if number == 1:
        fallbacks[0] = ""  # Experiment 0 is not allowed
    CM["previewer"].set_images(
        folder.parent / f"experiment_{number - 1:04d}/placeholder.png",
        folder.parent / f"experiment_{number:04d}/placeholder.png",
        folder.parent / f"experiment_{number + 1:04d}/placeholder.png",
        fallbacks=fallbacks
    )


async def _record():
    # Step 1: Save experiment
    if not _save_experiment():
        ui.notify("Failed to save experiment. Recording skipped.", color='negative')
        return

    # Step 2: Countdown
    if CM["checkbox_countdown"].value:
        # No click during countdown
        CM["button_record"].props("disable")

        # Set audio source and play
        if CM['select_voice'].value != "Silent":
            CM["audio_countdown"].set_source(f"assets/countdown/{CM['select_voice'].value}.mp3")
            CM["audio_countdown"].play()

        # Show countdown numbers
        CM["display_countdown"].style("display: block;")
        for word in ["3", "2", "1", "Go!"]:
            CM["display_countdown"].text = word
            await asyncio.sleep(1)
        CM["display_countdown"].style("display: none;")

    # Step 3: Start recording
    try:
        CM["button_record"].props(remove="disable").set_icon("stop_circle")
        _actual_record()
    finally:
        # Step 4: Restore UI
        CM["button_record"].set_icon("radio_button_checked")
        _on_change_experiment_number()


def _actual_record():
    """The actual recording process. Now just a placeholder."""
    lics = CM["select_lics"].value
    session = CM["select_session"].value
    number, folder = _get_experiment_folder()
    png_path = folder / "placeholder.png"
    plt.style.use('default')
    fig = plt.figure(dpi=200)
    fig.patch.set_facecolor("white")
    ax = fig.gca()
    ax.set_facecolor("white")
    ax.text(0, 0, f"Tracks obtained by\n\n"
                  f"{lics}\n{session}\n{_number_to_dir(number)}\n\n"
                  f"Number of channels: {len(CM['table_summary'].rows)}",
            ha="center", va="center",
            color="black", fontsize=20)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout(pad=0)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.02)
    fig.clf()
    plt.close(fig)


###################
# Local Interplay #
###################


def _go_previous():
    if CM["number_experiment"].value > 1:
        CM["number_experiment"].value -= 1


def _go_next():
    CM["number_experiment"].value += 1


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

        with MyUI.row():
            with ui.column().classes("flex-1"):
                # --- Experiment Number ---
                session_folder = DATA_DIR / lics / session
                number_init = _get_next_number(session_folder)
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
                CM["number_x"] = MyUI.number_int("Location X (cm)", value=-20,
                                                 on_change=_on_change_source_location, full=True)
                CM["number_y"] = MyUI.number_int("Location Y (cm)", value=0,
                                                 on_change=_on_change_source_location, full=True)

            with MyUI.cap_card("Source Increment", full=False):
                CM["number_inc_x"] = MyUI.number_int("Increment X (cm)", value=10, full=True)
                CM["number_inc_y"] = MyUI.number_int("Increment Y (cm)", value=0, full=True)

            #############
            # Operators #
            #############
            with MyUI.cap_card("Operators", full=False):
                with MyUI.row():
                    CM["input_computer_op"] = ui.input("Computer").classes('flex-1')
                    CM["input_source_op"] = ui.input("Source").classes('flex-1')
                with MyUI.row():
                    CM["input_protocol_op"] = ui.input("Protocol").classes('flex-1')
                    CM["input_others_op"] = ui.input("We are just here").classes('flex-1')

        #############
        # Auto Gain #
        #############
        with MyUI.cap_card("Auto Gain"):
            with MyUI.expansion(f"Gain Function"):
                with MyUI.row():
                    with ui.column().classes('w-2/3 gap-1'):
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
                            theme="githubDark" if GS.dark_mode else "githubLight",
                        ).classes("w-full").style('height: 160px;')
                    with ui.column().classes('flex-1'):
                        with ui.matplotlib(dpi=200, figsize=(4, 4)) \
                                .classes("w-full").figure as CM["figure_gain"]:  # noqa
                            CM["is_gain_valid"] = _plot_gain_curve(CM["figure_gain"])

        #################
        # Final summary #
        #################
        with MyUI.cap_card("Summary"):
            with MyUI.expansion(f"Experiment Details: 【 0 Channels 】") as CM["expansion_summary"]:
                with MyUI.row(gap=10):
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
                    ).classes('flex-1 q-table--col-auto-width')

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
                            />
                        </q-td>
                    ''')
                    CM["table_summary"].on('_on_change_summary_value', _on_change_summary_value)

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
                    CM["figure_summary"] = ui.matplotlib(dpi=200, figsize=(4, 4)).classes("flex-1").figure

            # --- Notes ---
            with MyUI.row():
                CM["input_pre_notes"] = ui.input("Pre-notes").classes('flex-1')
                CM["input_post_notes"] = ui.input("Post-notes") \
                    .classes('flex-1').props("readonly") \
                    .on("blur", _save_post_notes)

        # refresh
        _refresh_summary()

        ###########
        # Actions #
        ###########
        # Do not toggle time because we want static page during experiments
        # --- Create Time ---
        # CM["input_time"] = ui.input("Create Time").props("readonly").classes('w-full hidden')

        # --- Record ---
        with MyUI.cap_card("Record"):
            with MyUI.row().classes('items-center'):
                with ui.column().classes("flex-[5] justify-start"):
                    CM["checkbox_increment"] = ui.checkbox("Auto Increment Source", value=True) \
                        .tooltip("Auto-moves source when starting a new experiment. "
                                 "Only applies if the previous experiment is completed.").classes('w-full')

                    # Countdown
                    CM["checkbox_countdown"] = ui.checkbox("Countdown", value=False).classes('w-full')
                    files = sorted([audio[:-4] for audio in os.listdir('assets/countdown')
                                    if audio.endswith('.mp3')]) + ["Silent"]
                    CM["select_voice"] = ui.select(
                        files, value=np.random.choice(files[:-1])) \
                        .props('filled').classes('q-pt-none') \
                        .bind_enabled_from(CM["checkbox_countdown"], 'value').classes('w-3/4')
                    CM["audio_countdown"] = ui.audio("", autoplay=False, controls=False)
                    CM["display_countdown"] = ui.label("321").classes(
                        "fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 "
                        "text-primary text-[200px] font-bold z-50 pointer-events-none"
                    ).style("display: none;")

                with ui.column().classes("flex-[7] items-center justify-center"):
                    with ui.row().classes('w-full items-center justify-center'):
                        bg = MyUI.bg_color()
                        ft = MyUI.font_color()

                        CM["button_prev"] = ui.button(icon='chevron_left', on_click=_go_previous) \
                            .classes('h-24 text-5xl').props("flat") \
                            .style(f'background-color: {bg}; color: {ft}; border: none;')

                        CM["button_record"] = ui.button(icon='radio_button_checked', on_click=_record) \
                            .classes('h-24 text-5xl').props("flat") \
                            .style(f'background-color: {bg}; color: {ft}; border: none;')

                        CM["button_next"] = ui.button(icon='chevron_right', on_click=_go_next) \
                            .classes('h-24 text-5xl').props("flat") \
                            .style(f'background-color: {bg}; color: {ft}; border: none;')

                with ui.column().classes("flex-[5] justify-start"):
                    CM["text_final"] = ui.textarea().props('readonly borderless') \
                        .classes('text-2xl large-label h-24 self-end')

            with MyUI.expansion("Preview").classes("w-full"):
                CM["previewer"] = ThreeImageViewer()

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
