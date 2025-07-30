# 🎛️ EVO-16 Setup

## 🔌 Connection

If you're using more than eight channels, connect the **EVO-16** with the **EVO-SP8** using **optical cables** only (shown in green):

![Optical Connection](assets/doc/optical.png)

---

## 🎚️ Consistent Sampling Rate (SR)

When EVO-16 and EVO-SP8 are connected, both must operate at the **same sampling rate (SR)**.  
⚠️ The SR setup method differs between the two devices.

### 🖥️ Set SR on EVO-16 (macOS only)

1. Open **Audio MIDI Setup** from Launchpad:

   ![MIDI Setup Icon](assets/doc/midi_icon.png)

2. Select **EVO-16**, then set its SR to **44,100 Hz**:

   ![Set SR to 44100](assets/doc/midi.png)

---

### 🎛️ Set SR on EVO-SP8

1. On the EVO-SP8 hardware, press the main knob.
2. Navigate to **Sample Rate** and change the value to **44.1KHz**. See picture below.

   ![Verify SR](assets/doc/screen.png)

---

### ✅ Verify SR Consistency

On both devices, go to **Status** via the main knob.  
Ensure the display shows **44KHz** on each small screen.

---

## 🎙️ Mono Routing

Ensure channels are **not paired** for stereo.

In the image below:

- ✅ Channel 1 & 2 are **MONO** (correct)  
- ❌ Channel 3 & 4 are **STEREO** (incorrect) — click **STEREO** to decouple

![Mono Routing](assets/doc/mono.png)

Also, in `View → Show System Panel`, decouple all channels:

![System Panel](assets/doc/mono_sys.png)
