import argparse
import base64

import matplotlib.pyplot as plt
from nicegui import ui

from src.ui import GS, HELPS
from src.ui.about import initialize as about_initialize
from src.ui.experiment import initialize as experiment_initialize
from src.ui.lics import initialize as lics_initialize
from src.ui.session import initialize as session_initialize
from src.ui.utils import MyUI, show_help

# from src.ui.view import initialize as view_initialize

# ------------------------
# Parse command-line args
# ------------------------
parser = argparse.ArgumentParser(description="Run FieldUI.")
parser.add_argument('--theme', choices=['light', 'dark'], default='light', help='UI theme mode')
parser.add_argument('--port', type=int, default=8080, help='Port to run the app on')
parser.add_argument('--browser', action="store_true", help='Run in browser rather than native')
args = parser.parse_args()

# ------------------------
# Dark mode setup
# ------------------------
GS.dark_mode = (args.theme == 'dark')

# ------------------------
# Theme colors and CSS
# ------------------------
primary = MyUI.primary_color()
ui.colors(primary=primary, dark=MyUI.bg_color(), dark_page=MyUI.bg_color())
plt.style.use('dark_background' if GS.dark_mode else 'default')

ui.add_head_html(f'''
<style>
  .w-48 .q-tab__label {{
      font-size: 20px;
  }}
  .w-48 .q-tab__icon {{
      font-size: 36px;
      margin-bottom: 8px;
  }}
  .q-tab.q-tab--active .q-icon,
  .q-tab.q-tab--active .q-tab__label {{
      color: {primary} !important;
  }}
  .q-tab.q-tab--active .q-tab__indicator {{
      background-color: {primary} !important;
  }}
</style>
''')

# Global label size override
ui.add_head_html('''
<style>
    .large-label .q-field__label {
        font-size: 1.5rem;
        font-weight: 800;
    }
</style>
''')

# Global readonly visual override
ui.add_head_html('''
<style>
.q-field--readonly .q-field__native,
.q-field--readonly .q-field__input {
    color: %s !important;
}
</style>
''' % MyUI.gray_color())


# ------------------------
# Main tab bar (centered)
# ------------------------

def tab_with_help(name: str, icon: str, help_title: str, help_text: str, _tabs):
    with ui.row().classes('w-48 relative justify-center'):
        tab = ui.tab(name, icon=icon).classes('w-full')
        btn = ui.button(icon='help_outline',
                        on_click=lambda: show_help(help_title, help_text)
                        ).props('flat round size=sm dense').classes(
            'absolute bottom-[9px] right-[5px]'
        )
        btn.bind_visibility_from(_tabs, 'value', value=name)
    return tab


# --- Main tab bar ---
with ui.row().classes('justify-center w-full'):
    with ui.tabs() as tabs:
        t1 = tab_with_help('LICS', 'grass', 'LICS', HELPS["lics"], tabs)
        t2 = tab_with_help('SESSION', 'grain', 'SESSION', HELPS["session"], tabs)
        t3 = tab_with_help('EXPERIMENT', 'gavel', 'EXPERIMENT', HELPS["experiment"], tabs)
        t5 = ui.tab('ABOUT', icon='info_outline').classes('w-48')

# ------------------------
# Tab panels and content
# ------------------------
with ui.tab_panels(tabs, value="LICS").classes('w-full'):
    with ui.tab_panel(t1):
        lics_initialize()
    with ui.tab_panel(t2):
        session_initialize()
    with ui.tab_panel(t3):
        experiment_initialize()
    # with ui.tab_panel(t4):
    #     view_initialize()
    with ui.tab_panel(t5):
        about_initialize()

# ------------------------
# Favicon setup
# ------------------------
with open('assets/erp.jpeg', 'rb') as f:
    favicon_base64 = base64.b64encode(f.read()).decode()

# ------------------------
# Run app
# ------------------------
native = not args.browser
ui.run(
    title="Field UI",
    port=args.port,
    favicon=f'data:image/jpeg;base64,{favicon_base64}',
    dark=GS.dark_mode,
    reconnect_timeout=30,
    native=native,
    window_size=(1200, 800) if native else None
)
