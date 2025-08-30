import json
import os
import shutil
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
from countryinfo import CountryInfo
from nicegui import ui
from pycountry import countries, subdivisions

from src.ui import GS


class ControlManager:
    """A centralized manager for storing and updating UI controls by key."""

    def __init__(self):
        # Dictionary to store UI controls, accessible by key
        self.controls = {}

    def __setitem__(self, key, value):
        # Allows setting a control using bracket notation: manager[key] = control
        self.controls[key] = value

    def __getitem__(self, key):
        # Allows retrieving a control using bracket notation: manager[key]
        # Returns None if the key is not found
        return self.controls.get(key, None)

    def update(self, key, value=None, options=None, label=None, text=None,
               props=None, props_remove=False,
               classes=None, classes_remove=False, add_value_to_options=False):
        # Update various attributes of a control identified by key
        ctr = self.controls.get(key, None)
        if ctr is None:
            return

        # Update dropdown or selection options
        if options is not None:
            ctr.options = options

        if add_value_to_options and value not in ctr.options:
            ctr.options = ctr.options + [value]

        # Update the control's value
        if value is not None:
            ctr.value = value

        # Update the control's label
        if label is not None:
            ctr.label = label

        # Update the control's text
        if text is not None:
            ctr.text = text

        # Update or remove component properties
        if props is not None:
            if props_remove:
                ctr.props(remove=props)
            else:
                ctr.props(props)

        # Update or remove component CSS classes
        if classes is not None:
            if classes_remove:
                ctr.classes(remove=classes)
            else:
                ctr.classes(classes)


class CountryUtils:
    """Utility class for country-related operations such as lookup and geolocation."""

    # Mapping of country name to ISO alpha-2 code
    COUNTRIES_DICT = {
        c.name: c.alpha_2 for c in sorted(list(countries), key=lambda c: c.name)}  # noqa
    # List of all country names
    COUNTRIES_NAMES = list(COUNTRIES_DICT.keys())

    @staticmethod
    def get_subdivisions(country):
        """Return list of subdivision names for a given country name."""
        code = CountryUtils.COUNTRIES_DICT.get(country)
        if not code:
            return []
        return [s.name for s in subdivisions if s.country_code == code]  # noqa

    @staticmethod
    def get_latlon(country):
        """Return (lat, lon) tuple for a country, fallback to (0.0, 0.0) if unavailable."""
        try:
            info = CountryInfo(country).info()
            latlon = info.get("latlng", [])
            if len(latlon) == 2:
                return float(latlon[0]), float(latlon[1])
        except:  # noqa
            pass
        return 0.0, 0.0


def get_existing_sorted(directory: Path, prefix: str):
    """
    Return a list of subfolder names under the given directory,
    sorted by 'create_time' field from each subfolder's prefix_state.json (descending).
    """
    entries = []

    for subdir in directory.iterdir():
        if not subdir.is_dir():
            continue
        json_path = subdir / f"{prefix}_state.json"
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            create_time_str = meta.get("create_time")
            if not create_time_str:
                continue
            create_time = datetime.strptime(create_time_str, "%Y-%m-%dT%H:%M:%S")
            entries.append((subdir.name, create_time))
        except:  # noqa
            continue  # Skip folders with missing or invalid prefix_state.json

    entries.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in entries]


class MyPlot:
    """Plot Utils."""

    @staticmethod
    def error(fig, text):
        """Plot error figure with central red warning text."""
        ax = fig.gca()
        ax.text(0, 0, text, ha="center", va="center",
                color="red", fontsize=15)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout(pad=0)
        MyPlot.apply_dark(fig)

    @staticmethod
    def apply_margin(fig, xy_input):
        """Apply square margin padding to a 2D scatter plot."""
        ax = fig.gca()
        x_min, x_max = np.min(xy_input[:, 0]), np.max(xy_input[:, 0])
        y_min, y_max = np.min(xy_input[:, 1]), np.max(xy_input[:, 1])
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        half_range = max(x_max - x_min, y_max - y_min) / 2
        padding = 0.2 * half_range if half_range > 0 else 1.0
        ax.set_xlim(x_center - half_range - padding, x_center + half_range + padding)
        ax.set_ylim(y_center - half_range - padding, y_center + half_range + padding)
        ax.set_aspect('equal')

    @staticmethod
    def apply_dark(fig):
        """Apply dark color palette."""
        color = MyUI.bg_color()
        fig.patch.set_facecolor(color)
        fig.gca().set_facecolor(color)


class MyUI:
    """Custom UI components."""

    @staticmethod
    def bg_color(error=False):
        """Return background color."""
        if error:
            return "#8b0000" if GS.dark_mode else "#fee"
        return "#002147" if GS.dark_mode else "#FFFFFF"

    @staticmethod
    def font_color():
        """Return font color."""
        return "#FFFFFF" if GS.dark_mode else "#000000"

    @staticmethod
    def primary_color():
        """Return primary color."""
        return '#FFA500' if GS.dark_mode else '#2e7d32'

    @staticmethod
    def gray_color():
        return "#B0B0B0" if GS.dark_mode else "#666666"

    @staticmethod
    def row(gap=4):
        """ui.row() with full width and gap."""
        return ui.row().classes(f'w-full justify-between gap-{gap}')

    @staticmethod
    def expansion(text: str, value=False):
        """ui.expansion() with custom toggle icon."""

        def _on_change_expansion(e):
            """Update expansion icon on open/close."""
            e.sender.icon = "keyboard_arrow_up" if e.value else "keyboard_arrow_down"

        return ui.expansion(
            text,
            value=value,
            icon="keyboard_arrow_down",
            on_value_change=_on_change_expansion,
        ).classes("w-full").props("expand-icon=none")

    @staticmethod
    @contextmanager
    def cap_card(caption: str, full=True, highlight=False, height_px=None):
        """Card with a floating caption."""
        ft_color = MyUI.primary_color() if highlight else MyUI.gray_color()
        card_classes = 'w-full' if full else 'flex-1'
        if height_px is not None:
            card_classes += f' h-[{height_px}px]'
        card_style = 'border-color: var(--q-primary);' if highlight else ''
        label_classes = 'absolute -top-2.5 left-4 q-px-sm'
        if highlight:
            label_classes += ' font-bold'

        with ui.card().props('flat bordered').classes(card_classes).style(card_style) as card:
            ui.label(caption).classes(label_classes).style(
                f'background-color: {MyUI.bg_color()}; color: {ft_color};'
            )
            yield card

    @staticmethod
    def number_int(label, value=None, min=None, max=None, on_change=None, full=True):  # noqa
        """Integer-only number input with auto-round on blur."""

        def _round_to_int(e):
            try:
                val = float(e.sender.value)
                val = int(round(val))
                e.sender.value = val
            except:  # noqa
                e.sender.value = e.sender.min or 0  # fallback

        return ui.number(
            label,
            value=value,
            min=min,
            max=max,
            on_change=(lambda e: (_round_to_int(e), on_change and on_change(e))[1]),
        ).on('blur', _round_to_int) \
            .classes('w-full' if full else 'flex-1')

    @staticmethod
    def checkbox(text, value=None, on_change=None, full=True):
        """Checkbox aligned to input height (56px)."""
        with ui.row().style('align-items: center; height: 56px;'):
            checkbox = ui.checkbox(
                text,
                value=value,
                on_change=on_change
            ).classes('w-full' if full else 'flex-1')
        return checkbox


class CallbackBlocker:
    def __init__(self):
        self._block_callback = False

    @contextmanager
    def block(self):
        """Temporarily block callbacks during UI updates."""
        self._block_callback = True
        try:
            yield
        finally:
            self._block_callback = False

    def blocking(self):
        """Check if callbacks are currently blocked."""
        return self._block_callback


class ThreeImageViewer:
    def __init__(self, height='320px', ratio=(2, 3, 2), gap=10):
        with ui.row().classes('items-center w-full') \
                .style(f'height: {height}; gap: {gap}px; perspective: 1000px;'):
            self.left_img = ui.image('').style('''
                width: 100%; height: 100%;
                transform: rotateY(24deg);
                transform-origin: left;
                opacity: 0.7;
            ''').classes(f'flex-[{ratio[0]}]').props('fit=contain')
            self.left_icon = ui.icon('chevron_left') \
                .classes('absolute left-2 top-1/2 -translate-y-1/2 '
                         'text-6xl pointer-events-none text-primary')

            self.middle_img = ui.image('').style('''
                width: 100%; height: 100%;
            ''').classes(f'flex-[{ratio[1]}]').props('fit=contain')

            self.right_img = ui.image('').style('''
                width: 100%; height: 100%;
                transform: rotateY(-24deg);
                transform-origin: right;
                opacity: 0.7;
            ''').classes(f'flex-[{ratio[2]}]').props('fit=contain')
            self.right_icon = ui.icon('chevron_right') \
                .classes('absolute right-2 top-1/2 -translate-y-1/2 '
                         'text-6xl pointer-events-none text-primary')

    def set_images(self, left, middle, right, fallbacks, number, number_min, number_max):
        """Set image sources. If a path doesn't exist, fallbacks are used."""
        self.left_img.set_source(left if Path(left).exists() else fallbacks[0])
        self.middle_img.set_source(middle if Path(middle).exists() else fallbacks[1])
        self.right_img.set_source(right if Path(right).exists() else fallbacks[2])
        if number > number_min:
            self.left_icon.style('visibility: visible')
        else:
            self.left_icon.style('visibility: hidden')
        if number < number_max:
            self.right_icon.style('visibility: visible')
        else:
            self.right_icon.style('visibility: hidden')


def _detect_snuffler():
    """
    Cross-platform detection of Snuffler executable.
    """
    # Step 1: Try system PATH
    path = shutil.which("snuffler")
    if path:
        return path

    # Pick executable name based on platform
    exe_name = "snuffler.exe" if os.name == "nt" else "snuffler"

    # Step 2: Check common conda base locations
    candidate_dirs = [
        Path.home() / "anaconda3",
        Path.home() / "miniconda3",
        Path.home() / "mambaforge",
        Path("/opt/anaconda3"),
    ]

    for base in candidate_dirs:
        envs_dir = base / "envs"
        if envs_dir.exists():
            for env in envs_dir.iterdir():
                if os.name == "nt":
                    snuffler_path = env / "Scripts" / exe_name
                else:
                    snuffler_path = env / "bin" / exe_name
                if snuffler_path.exists() and os.access(snuffler_path, os.X_OK):
                    return str(snuffler_path)

    # Step 3: Check user-specific envs (~/.conda/envs)
    user_envs = Path.home() / ".conda" / "envs"
    if user_envs.exists():
        for env in user_envs.iterdir():
            if os.name == "nt":
                snuffler_path = env / "Scripts" / exe_name
            else:
                snuffler_path = env / "bin" / exe_name
            if snuffler_path.exists() and os.access(snuffler_path, os.X_OK):
                return str(snuffler_path)

    # Step 4: Fallback
    return exe_name
