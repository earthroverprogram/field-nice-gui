import threading
import time

import numpy as np
import sounddevice as sd

STATIC_DEVICE_MAP = {
    "Built-in Mic": lambda phys_name: any(
        known_phys.lower() in phys_name.lower()
        for known_phys in [
            "MacBook Air Microphone",
            "MacBook Pro Microphone",
            "iMac Microphone",
            "Internal Microphone",
            "Built-in Microphone",
            "MacBook Microphone"
        ]
    ),
    "EVO-16": lambda phys_name: "evo" in phys_name.lower() and "16" in phys_name.lower(),
    "Scarlett-2i2": lambda phys_name: "scarlett" in phys_name.lower() and "2i2" in phys_name.lower(),
    "Digiface": lambda phys_name: "digiface" in phys_name.lower(),
    "Dummy": lambda phys_name: phys_name == "Dummy"
}

DUMMY_CHANNELS = 32


class Datalogger:
    def __init__(self):
        self.stop_flag = False
        self.buffer = []
        self.thread = None
        self.stream = None
        self.active_channels = []
        self.mode = None
        self.on_data = None
        self.datatype = None
        self.original_channels = None
        self.invalid_channels = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_streaming()

    @staticmethod
    def get_physical_devices():
        """
        Return a dict of available physical devices.
        Format: {physical_name: num_input_channels}
        """
        # Detection
        try:
            # Without re-initialization, query_devices won't update
            sd._terminate()  # noqa
            sd._initialize()  # noqa
        except:  # noqa
            pass
        devices = sd.query_devices()
        phys_devices = {
            dev["name"]: dev["max_input_channels"]
            for dev in devices if dev["max_input_channels"] > 0
        }
        # Add Dummy
        phys_devices["Dummy"] = DUMMY_CHANNELS
        return phys_devices

    @staticmethod
    def get_logical_devices():
        """
        Return a dict of available input devices using STATIC_DEVICE_MAP.
        Format: {logical_name: num_input_channels}
        Matching logic:
          - exactly one match → bind and return real channel count
          - zero or multiple matches → return 0
        """
        # Get available devices
        phys_devices = Datalogger.get_physical_devices()

        # Match logical names
        result = {}
        for logical_name, matcher in (STATIC_DEVICE_MAP.items()):
            matched = [phys_name for phys_name in phys_devices if matcher(phys_name)]
            if len(matched) == 1:
                # Exact match
                result[logical_name] = {
                    "physical_name": matched[0],
                    "n_chs": phys_devices[matched[0]]
                }
            else:
                # No match or more than one matches: discard
                result[logical_name] = {
                    "physical_name": "",
                    "n_chs": 0
                }
        return result

    def _start_streaming_dummy(self, datatype, samplerate, mode, on_data=None):
        self.mode = mode
        self.on_data = on_data
        self.stop_flag = False
        self.buffer = [] if mode == 'record' else None

        dtype = np.dtype(datatype)
        blocksize = 512
        interval = blocksize / samplerate
        rng = np.random.default_rng()

        def dummy_loop():
            while not self.stop_flag:
                # Generate band-limited random signal
                base = rng.normal(0, 0.2, size=(blocksize, DUMMY_CHANNELS))
                base = np.cumsum(base, axis=0)  # Optional: integrate to get smoother appearance

                # Optional: slight channel offset
                scale = 0.5  # keep sine wave within ±0.5 for float types
                if np.issubdtype(dtype, np.integer):
                    scale = 0.5 * np.iinfo(dtype).max

                for ch in self.active_channels:
                    wave = np.sin(2 * np.pi * (ch + 1) * np.arange(blocksize) / samplerate)
                    base[:, ch] = (scale * (base[:, ch] + wave)).astype(dtype)

                selected = base[:, self.active_channels]
                if self.mode == 'record':
                    self.buffer.append(selected)
                elif self.mode == 'monitor' and self.on_data:
                    self.on_data(selected)

                time.sleep(interval)

        self.thread = threading.Thread(target=dummy_loop)
        self.thread.start()

    def _start_streaming(self, logical_name, channel_list,
                         datatype, samplerate, mode, ignore_invalid_channels=True, on_data=None):
        self.datatype = datatype
        self.mode = mode
        self.on_data = on_data
        self.stop_flag = False
        self.buffer = [] if mode == 'record' else None
        self.active_channels = [ch - 1 for ch in channel_list]
        self.original_channels = self.active_channels.copy()
        self.invalid_channels = []

        info = self.get_logical_devices().get(logical_name)
        if not info or not info["physical_name"] or info["n_chs"] == 0:
            raise ValueError(f"Logical device '{logical_name}' not mapped to any physical device.")

        physical_name = info["physical_name"]
        max_channels = info["n_chs"]
        self.invalid_channels = [ch for ch in self.active_channels if ch >= max_channels]

        if self.invalid_channels:
            if ignore_invalid_channels:
                self.active_channels = [ch for ch in self.active_channels if ch < max_channels]
            else:
                raise ValueError(
                    f"Device '{physical_name}' does not support channels: "
                    f"{[ch + 1 for ch in self.invalid_channels]}")

        if physical_name == "Dummy":
            self._start_streaming_dummy(datatype, samplerate, mode, on_data)
            return

        def callback(indata, frames, time_info, status):  # noqa
            if self.stop_flag:
                raise sd.CallbackStop()
            selected = indata[:, self.active_channels].copy()
            if self.mode == 'record':
                self.buffer.append(selected)
            elif self.mode == 'monitor' and self.on_data:
                self.on_data(selected)

        self.stream = sd.InputStream(
            samplerate=samplerate,
            channels=max(self.active_channels) + 1 if self.active_channels else 1,
            device=physical_name,
            callback=callback,
            dtype=datatype
        )

        self.thread = threading.Thread(target=self._stream_runner)
        self.thread.start()

    def stop_streaming(self):
        self.stop_flag = True
        if self.thread:
            self.thread.join()
            self.thread = None

        if self.mode == 'record' and self.buffer:
            recorded = np.concatenate(self.buffer, axis=0)
            n_samples = recorded.shape[0]
            full_data = np.zeros((n_samples, len(self.original_channels)), dtype=self.datatype)
            ch_map = {ch: i for i, ch in enumerate(self.original_channels)}
            valid_map = {ch: i for i, ch in enumerate(self.active_channels)}
            for ch in self.original_channels:
                if ch in valid_map:
                    full_data[:, ch_map[ch]] = recorded[:, valid_map[ch]]
                else:
                    # invalid channel remains zero
                    pass
            return full_data
        else:
            return np.empty((0, len(getattr(self, 'original_channels', []))), dtype=self.datatype)

    def start_recording(self, logical_name, channel_list, datatype="float32", samplerate=44100):
        """
        Start recording audio to memory. Use stop_recording() to stop and retrieve result.
        """
        self._start_streaming(logical_name, channel_list, datatype, samplerate, mode='record')

    def start_monitoring(self, logical_name, channel_list, datatype="float32", samplerate=44100, on_data=None):
        """
        Start real-time monitoring without recording.
        on_data: callback function to receive audio blocks
        """
        self._start_streaming(logical_name, channel_list, datatype, samplerate, mode='monitor', on_data=on_data)

    def _stream_runner(self):
        """
        Background runner for the audio stream.
        """
        with self.stream:
            while not self.stop_flag:
                sd.sleep(100)

    def set_preamp_gain(self, device, channel_gain_dict):
        """
        Set preamp gain at the physical level.
        """
        pass
