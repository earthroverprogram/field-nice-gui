# 🌱 Field UI — A “Nice” One

---

## 🐍 Install Anaconda

Field UI is designed to run inside an **Anaconda environment**.  
If you don’t already have Anaconda installed:

- Download from: [https://www.anaconda.com/download](https://www.anaconda.com/download)  
- Follow the installer instructions for your operating system (Windows, macOS, or Linux).  

Once installed, you can proceed with the steps below.

---

## 📦 Install Field-UI

All commands below must be run in a **terminal with Anaconda available**.  
- On **Windows**, use the **Anaconda Prompt** (not PowerShell or CMD).  
- On **macOS / Linux**, use your regular terminal.

```bash
# Create a new conda environment with Python 3.10
conda create --name field-nice-gui python=3.10

# Activate the environment
conda activate field-nice-gui

# Clone the project repository from GitHub
# (Skip this step if you downloaded the zip)
git clone https://github.com/earthroverprogram/field-nice-gui

# Change into the project folder
cd field-nice-gui

# Install required packages
pip install -r requirements.txt
```

> ⚠️ **Important Compatibility Note**  
> - **NiceGUI** requires **NumPy 2**  
> - **Pyrocko (Snuffler)** requires **NumPy 1**  
> These **cannot** be installed in the same environment.

- If **Snuffler** is already installed, you don’t need to reinstall it.  
  The UI will try to auto-detect the `snuffler` executable (e.g. `/opt/anaconda3/envs/some_env/bin/snuffler`).  
  If detection fails, you can manually specify its path in the UI.

- If **Snuffler** is not installed, you must create a **separate conda environment** for it, e.g.:

  ```bash
  conda create --name snuffler-env python=3.10
  conda activate snuffler-env
  pip install pyrocko
  pip install PyQt5
  ```

---

## ⚙️ Config the App
You can edit `data_dir` in `config.json` to separate data and code folders.

---

## ▶️ Run the App

Make sure you are in the correct environment and inside the `field-nice-gui` folder.  
- On **Windows**, run these commands in **Anaconda Prompt**.  
- On **macOS / Linux**, run them in your regular terminal.

Basic usage:

```bash
conda activate field-nice-gui
python app.py
```

Full CLI options:

```
usage: app.py [-h] [--theme {light,dark}] [--port PORT] [--browser]

Run FieldUI.

options:
  -h, --help            Show this help message and exit
  --theme {light,dark}  Set UI theme mode
  --port PORT           Port to run the app on
  --browser             Launch in browser instead of native window
```

---

## 🎛️ EVO-16 Integration

To ensure **EVO-16** works correctly, follow the setup guide in:  
[📘 EVO-16 Setup Guide](README_EVO16.md)
