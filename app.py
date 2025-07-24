import base64
import sys

import matplotlib.pyplot as plt
from nicegui import ui

from src.ui import GS
from src.ui.about import initialize as about_initialize
from src.ui.experiment import initialize as experiment_initialize
from src.ui.lics import initialize as lics_initialize
from src.ui.session import initialize as session_initialize
from src.ui.utils import MyUI
from src.ui.view import initialize as view_initialize

# ------------------------
# Static dark mode setup
# ------------------------
GS.dark_mode = 'dark' in " ".join(sys.argv).lower()

# ------------------------
# Theme colors and CSS
# ------------------------
primary = MyUI.primary_color()
ui.colors(primary=primary)
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
with ui.row().classes('justify-center w-full'):
    with ui.tabs() as tabs:
        t1 = ui.tab('LICS', icon='grass').classes('w-48')
        t2 = ui.tab('SESSION', icon='grain').classes('w-48')
        t3 = ui.tab('EXPERIMENT', icon='gavel').classes('w-48')
        t4 = ui.tab('VIEW', icon='troubleshoot').classes('w-48')
        t5 = ui.tab('ABOUT', icon='info_outline').classes('w-48')
tabs.value = t1

# ------------------------
# Tab panels and content
# ------------------------
with ui.tab_panels(tabs, value=t1).classes('w-full'):
    with ui.tab_panel(t1):
        lics_initialize()
    with ui.tab_panel(t2):
        session_initialize()
    with ui.tab_panel(t3):
        experiment_initialize()
    with ui.tab_panel(t4):
        view_initialize()
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
ui.run(
    title="Field UI",
    favicon=f'data:image/jpeg;base64,{favicon_base64}',
    dark=GS.dark_mode
)
