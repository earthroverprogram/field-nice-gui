import os

# Set ASIO environment variable before importing sounddevice
os.environ["SD_ENABLE_ASIO"] = "1"

import threading
import time

import numpy as np
import sounddevice as sd

STATIC_DEVICE_MAP = {
    "Dummy": lambda phys_name, api_name: phys_name == "Dummy",

    "EVO-16": lambda phys_name, api_name:
    ("evo16" in phys_name.lower() and "core audio" in api_name.lower()) or
    ("audient usb" in phys_name.lower() and "asio" in api_name.lower()),

    "Scarlett-2i2": lambda phys_name, api_name:
    "scarlett 2i2" in phys_name.lower() and "core audio" in api_name.lower(),

    "Digiface": lambda phys_name, api_name:
    "digiface" in phys_name.lower() and "core audio" in api_name.lower()
}

DUMMY_CHANNELS = 32
DUMMY_SR = 44100


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
        self.last_exception = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_streaming()

    @staticmethod
    def _get_preferred_hostapis():
        """
        Return list of preferred host API names
        - Core Audio (macOS)
        - ASIO (Windows) -- for best compatibility, we keep the minimum
        - ALSA (Linux)
        """
        apis = sd.query_hostapis()

        # macOS
        for api in apis:
            if 'core audio' in api['name'].lower():
                return [api['name']]

        # Windows
        for api in apis:
            if 'asio' in api['name'].lower():
                return [api['name']]

        # Linux
        for api in apis:
            if 'alsa' in api['name'].lower():
                return [api['name']]

        return []

    @staticmethod
    def get_physical_devices():
        """
        Return a dict of available physical devices.
        Format: {device_key: {physical_index, physical_name, api_name, n_chs}}
        Only considers preferred host APIs to avoid duplicates.
        """
        # Detection
        try:
            # Without re-initialization, query_devices won't update
            sd._terminate()  # noqa
            sd._initialize()  # noqa
        except:  # noqa
            pass

        # Get preferred host APIs
        preferred_apis = Datalogger._get_preferred_hostapis()
        if not preferred_apis:
            print("Warning: No preferred host APIs found")
            return {}

        # Query from sd
        devices = sd.query_devices()
        apis = sd.query_hostapis()

        phys_devices = {}
        for idx, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                # Get the host API name for this device
                api_index = dev['hostapi']
                api_name = apis[api_index]['name'] if api_index < len(apis) else 'Unknown'

                # Skip if not in preferred APIs
                if api_name not in preferred_apis:
                    continue

                name = dev["name"]
                if name == "Dummy":
                    continue  # we'll add Dummy later

                try:
                    # # Quick open test
                    # with sd.InputStream(device=idx, channels=1,
                    #                     samplerate=dev["default_samplerate"], dtype="float32",
                    #                     blocksize=64):
                    #     sd.sleep(20)  # short test, ms

                    # Create unique key
                    device_key = f"{idx} - {name}"
                    phys_devices[device_key] = {
                        "physical_index": idx,
                        "physical_name": name,
                        "api_name": api_name,
                        "n_chs": dev["max_input_channels"],
                        "default_sr": dev["default_samplerate"]
                    }
                except:  # noqa
                    # skip unusable device
                    continue

        # Add Dummy
        phys_devices["Dummy"] = {
            "physical_index": -1,  # Special index for dummy
            "physical_name": "Dummy",
            "api_name": "Dummy",
            "n_chs": DUMMY_CHANNELS,
            "default_sr": DUMMY_SR
        }

        return phys_devices

    @staticmethod
    def get_logical_devices():
        """
        Return a dict of available input devices using STATIC_DEVICE_MAP.
        Format: {logical_name: {physical_index, physical_name, api_name, n_chs}}
        Matching logic:
          - exactly one match → bind and return real info
          - zero or multiple matches → return empty info
        """
        # Get available devices
        phys_devices = Datalogger.get_physical_devices()

        # Match logical names
        result = {}
        matched_physical_devices = []

        for logical_name, matcher in STATIC_DEVICE_MAP.items():
            # Match against both physical_name and api_name
            matched = [key for key, info in phys_devices.items()
                       if matcher(info["physical_name"], info["api_name"])]

            if len(matched) == 1:
                # Exact match
                device_key = matched[0]
                result[logical_name] = phys_devices[device_key].copy()
                matched_physical_devices.append(device_key)
            else:
                # No match or more than one matches: discard
                result[logical_name] = {
                    "physical_index": -1,
                    "physical_name": "",
                    "api_name": "",
                    "n_chs": 0,
                    "default_sr": DUMMY_SR
                }

        # Add unmatched physical devices
        for device_key, device_info in phys_devices.items():
            if device_key not in matched_physical_devices:
                # Use the device_key as the logical name for unmatched devices
                result[device_key] = device_info.copy()

        return result

    def _prepare_streaming_dummy(self, datatype, samplerate, mode, on_data=None):
        self.mode = mode
        self.on_data = on_data
        self.stop_flag = False
        self.buffer = [] if mode == 'record' else None

        dtype = np.dtype(datatype)
        blocksize = 512
        interval = blocksize / samplerate
        rng = np.random.default_rng()

        def dummy_loop():
            try:
                while not self.stop_flag:
                    # Generate band-limited random signal
                    base = rng.normal(0, 0.2, size=(blocksize, DUMMY_CHANNELS))
                    base = np.cumsum(base, axis=0)  # Optional: integrate to get smoother appearance

                    # Optional: slight channel offset
                    scale = 0.5  # keep sine wave within ±0.5 for float types
                    if np.issubdtype(dtype, np.integer):
                        scale = 0.1 * np.iinfo(dtype).max

                    for ch in self.active_channels:
                        wave = np.sin(2 * np.pi * (ch + 1) * np.arange(blocksize) / samplerate)
                        base[:, ch] = scale * (base[:, ch] + wave)

                    if np.issubdtype(dtype, np.integer):
                        info = np.iinfo(dtype)
                        selected = np.clip(base[:, self.active_channels], info.min, info.max).astype(dtype)
                    else:
                        selected = base[:, self.active_channels].astype(dtype)

                    if self.mode == 'record':
                        self.buffer.append(selected)
                    elif self.mode == 'monitor' and self.on_data:
                        self.on_data(selected)

                    time.sleep(interval)
            except Exception as e:
                self.last_exception = e
                self.stop_flag = True  # kill the loop gracefully

        return threading.Thread(target=dummy_loop)

    def _prepare_streaming(self, logical_name, channel_list,
                           datatype, samplerate, mode, ignore_invalid_channels=True, on_data=None):
        if datatype == "int24":
            datatype = "int32"  # numpy does not support int24

        self.datatype = datatype
        self.mode = mode
        self.on_data = on_data
        self.stop_flag = False
        self.buffer = [] if mode == 'record' else None
        self.active_channels = [int(ch) - 1 for ch in channel_list]
        self.original_channels = self.active_channels.copy()
        self.invalid_channels = []

        info = self.get_logical_devices().get(logical_name)
        if not info or not info["physical_name"] or info["n_chs"] == 0:
            raise ValueError(f"Logical device '{logical_name}' not mapped to any physical device.")

        physical_index = info["physical_index"]
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
            return self._prepare_streaming_dummy(datatype, samplerate, mode, on_data)

        def callback(indata, frames, time_info, status):  # noqa
            try:
                selected = indata[:, self.active_channels].copy()
                if self.mode == 'record':
                    self.buffer.append(selected)
                elif self.mode == 'monitor' and self.on_data:
                    self.on_data(selected)
            except Exception as e:
                self.last_exception = e
                self.stop_flag = True  # kill the loop gracefully
                raise sd.CallbackStop()

        # Use physical_index instead of physical_name to avoid ambiguity
        device_param = physical_index if physical_index >= 0 else physical_name

        self.stream = sd.InputStream(
            samplerate=samplerate,
            channels=max(self.active_channels) + 1 if self.active_channels else 1,
            device=device_param,
            callback=callback,
            dtype=datatype,
            latency=0.1
        )

        return threading.Thread(target=self._stream_runner)

    def stop_streaming(self):
        """Signal the background thread to stop and wait for it to exit.

        IMPORTANT:
        - Do NOT call abort()/close() on the PortAudio stream here because the stream
          context is owned by the background thread (_stream_runner). Closing from a
          different thread can trigger AUHAL err=-50 on macOS and/or double-close.
        """
        self.stop_flag = True

        # Join the background thread so `_stream_runner` can exit its `with self.stream:`
        # cleanly and close the stream from the same thread that opened it.
        if self.thread:
            self.thread.join(timeout=1.5)
            if self.thread.is_alive():
                print("Warning: stream thread did not exit cleanly")
            self.thread = None

        # At this point, the stream context should already be closed by `_stream_runner`.
        # Ensure we drop the reference to avoid accidental reuse.
        self.stream = None

        # If we were recording, consolidate buffered blocks into the full data array.
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

    def prepare_recording(self, logical_name, channel_list, datatype="float32", samplerate=44100):
        """
        Prepare recording audio to memory. Use stop_recording() to stop and retrieve result.
        """
        self.thread = self._prepare_streaming(logical_name, channel_list, datatype, samplerate, mode='record')

    def start_recording(self):
        """
        Start recording.
        Must call prepare_recording() first.
        """
        if self.thread is None:
            raise RuntimeError("Recording thread not prepared. Call prepare_recording() first.")
        if self.thread.is_alive():
            raise RuntimeError("Recording thread already running.")
        self.thread.start()

    def start_monitoring(self, logical_name, channel_list, datatype="float32", samplerate=44100, on_data=None):
        """
        Start real-time monitoring without recording.
        on_data: callback function to receive audio blocks
        """
        self.thread = self._prepare_streaming(logical_name, channel_list, datatype, samplerate, mode='monitor',
                                              on_data=on_data)
        self.thread.start()

    def _stream_runner(self):
        """
        Background runner for the audio stream.
        Owns the stream context: open, keep alive, and close on exit.
        """
        try:
            with self.stream:
                while not self.stop_flag:
                    sd.sleep(100)
        finally:
            # Ensure the stream is closed from this same thread.
            # The `with` context should already have closed it, but this guards against
            # partial initialization or exceptions before entering the context.
            try:
                if self.stream is not None:
                    # Avoid calling .abort()/.close() from other threads.
                    # .close() here is idempotent and safe if already closed by context.
                    try:
                        self.stream.close()
                    except:  # noqa
                        pass
            finally:
                self.stream = None
