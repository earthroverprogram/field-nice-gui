from pathlib import Path


class State:
    """Global state container to help cross-tab communications."""

    def __init__(self):
        # Currently selected license identifier (default: <NEW>)
        self.selected_lics = "<NEW>"

        # Currently selected session identifier (default: <NEW>)
        self.selected_session = "<NEW>"

        # Dark mode
        self.dark_mode = False

    def get_selected(self):
        """Return a tuple of (selected_lics, selected_session)."""
        return self.selected_lics, self.selected_session


# Global instance to be shared across modules
GS = State()

# Global path to the data directory
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)  # Ensure the data directory exists
