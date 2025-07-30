# 🌱 Field UI — A “Nice” One

## 📦 Installation

```bash
pip install -r requirements.txt
```

⚠️ **Important**:  
NiceGUI requires **NumPy 2**, while **Pyrocko (Snuffler)** is only compatible with **NumPy 1**.  
To use both:

- You **must** install **Pyrocko** in a **separate conda environment** (e.g. `pip install pyrocko`).
- If **Snuffler** is already working, you don’t need to reinstall it — just install **Field UI** in another environment.
- The UI will automatically detect your `snuffler` executable (e.g. `/opt/anaconda3/envs/some_env/bin/snuffler`).  
  If not, you can manually specify its path in the UI.

---

## ▶️ Run the App

Basic usage:

```bash
python app.py
```

Full CLI options:

```bash
python app.py [-h] [--theme {light,dark}] [--port PORT]
```

---

## 🎛️ EVO-16 Integration

To ensure **EVO-16** works properly, please follow the setup guide in  
[📘 EVO-16 Manual](README_EVO16.md)
