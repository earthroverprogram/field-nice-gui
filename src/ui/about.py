from nicegui import ui

try:
    from src import VERSION
except ImportError:
    VERSION = "0.0"


def initialize():
    with ui.column().classes('w-full items-center justify-center mt-12 gap-4'):
        # Title + Version
        with ui.row().classes("items-end"):
            ui.label('Field UI').classes('text-4xl font-bold')
            ui.label(f'v{VERSION}').classes('text-xl font-bold')

        # Logo
        ui.image('assets/erp.png').classes('w-48 rounded-xl')

        # Copyright
        ui.label('© 2025 All rights reserved.').classes('text-gray-500')
