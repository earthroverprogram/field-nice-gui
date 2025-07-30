# 🌱 Field UI — A “Nice” One

## 📦 Installation

```bash
pip install -r requirements.txt
```

⚠️ **Important**:  
NiceGUI requires **NumPy 2**, while **Pyrocko (Snuffler)** is only compatible with **NumPy 1**.  
To use both:

- Install **Pyrocko** in a **separate conda environment**.
- The UI will automatically detect your `snuffler` executable.

## ▶️ Run the App

Basic usage:

```bash
python app.py
```

Full CLI options:

```bash
python app.py [-h] [--theme {light,dark}] [--port PORT]
```

## 🎛️ EVO-16 Integration

To ensure **EVO-16** functions properly, please follow the setup guide in [📘 EVO-16 Manual](README_EVO16.md)
