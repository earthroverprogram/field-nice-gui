import threading

import numpy as np
import sounddevice as sd


class Datalogger:
    STATIC_DEVICE_MAP = {
        "Built-in Mic": (lambda name: any(
            key.lower() in name.lower()
            for key in [
                "MacBook Air Microphone",
                "MacBook Pro Microphone",
                "iMac Microphone",
                "Internal Microphone",
                "Built-in Microphone",
                "MacBook Microphone",
            ]
        ), 0),
        "EVO-16": (lambda name: "evo" in name.lower() and "16" in name.lower(), 0),
        "Scarlett-2i2": (lambda name: "scarlett" in name.lower() and "2i2" in name.lower(), 0),
        "Digiface": (lambda name: "digiface" in name.lower(), 0),
        "Dummy": (lambda name: name == "Dummy", 64)
    }

    def __init__(self):
        self.stop_flag = False
        self.buffer = []  # Only used in 'record' mode
        self.thread = None
        self.stream = None
        self.active_channels = []  # 0-based list of selected channels
        self.mode = None  # 'record' or 'monitor'
        self.on_data = None  # callback function for monitor mode

    @staticmethod
    def get_devices():
        """
        Return a dict of available input devices using STATIC_DEVICE_MAP.
        Format: {logical_name: num_input_channels}
        Matching logic:
          - exactly one match → bind and return real channel count
          - zero or multiple matches → return 0
        """
        # Get available devices
        devices = sd.query_devices()
        available = {
            dev["name"]: dev["max_input_channels"]
            for dev in devices
            if dev["max_input_channels"] > 0
        }


        # Match logical names
        result = {}
        for logical_name, (matcher, default_channels) in Datalogger.STATIC_DEVICE_MAP.items():
            matched = [name for name in available if matcher(name)]
            if len(matched) == 1:
                result[logical_name] = available[matched[0]]
            else:
                result[logical_name] = default_channels

        return result

    def set_gain(self, device, channel_gain_dict):
        """
        Store user-defined gain values per 1-based channel.
        These are not applied to the recorded data.
        """
        pass

    def _start_streaming(self, device, channel_list, samplerate, mode, on_data=None):
        """
        Internal shared method to start a recording or monitoring stream.

        Parameters:
            device: name (str) or index (int)
            channel_list: list of 1-based channel numbers to use
            samplerate: sampling rate in Hz
            mode: 'record' or 'monitor'
            on_data: callback to handle live blocks (only used in 'monitor' mode)
        """
        devices = sd.query_devices()

        # Validate device
        if isinstance(device, str):
            if device not in [d['name'] for d in devices]:
                raise ValueError(f"Device '{device}' not found.")
        elif isinstance(device, int):
            if not (0 <= device < len(devices)):
                raise ValueError(f"Device index {device} is invalid.")
        else:
            raise TypeError("Device must be a string (name) or integer (index).")

        if not channel_list:
            raise ValueError("channel_list must not be empty.")

        self.active_channels = [ch - 1 for ch in channel_list]
        device_info = sd.query_devices(device, 'input')
        max_channels = device_info['max_input_channels']
        for ch in self.active_channels:
            if ch < 0 or ch >= max_channels:
                raise ValueError(f"Channel {ch + 1} is out of range for device '{device}'.")

        self.stop_flag = False
        self.mode = mode
        self.on_data = on_data
        self.buffer = [] if mode == 'record' else None

        def callback(indata, frames, time, status):
            if self.stop_flag:
                raise sd.CallbackStop()
            selected = indata[:, self.active_channels].copy()

            if self.mode == 'record':
                self.buffer.append(selected)
            elif self.mode == 'monitor' and self.on_data:
                self.on_data(selected)

        self.stream = sd.InputStream(
            samplerate=samplerate,
            channels=max(self.active_channels) + 1,
            device=device,
            callback=callback,
            dtype='float32'
        )

        self.thread = threading.Thread(target=self._stream_runner)
        self.thread.start()

    def start_recording(self, device, channel_list, samplerate=44100):
        """
        Start recording audio to memory. Use stop_recording() to stop and retrieve result.
        """
        self._start_streaming(device, channel_list, samplerate, mode='record')

    def start_monitoring(self, device, channel_list, samplerate=44100, on_data=None):
        """
        Start real-time monitoring without recording.
        on_data: callback function to receive audio blocks
        """
        self._start_streaming(device, channel_list, samplerate, mode='monitor', on_data=on_data)

    def _stream_runner(self):
        """
        Background runner for the audio stream.
        """
        with self.stream:
            while not self.stop_flag:
                sd.sleep(100)

    def stop_streaming(self):
        """
        Stop stream and return recorded audio if in 'record' mode.
        Returns:
            numpy array of shape (samples, len(channel_list)) or empty array
        """
        self.stop_flag = True
        if self.thread:
            self.thread.join()

        if self.mode == 'record' and self.buffer:
            return np.concatenate(self.buffer, axis=0)
        else:
            return np.empty((0, len(self.active_channels)), dtype='float32')
