# 🌱 Field UI — A “Nice” One

## 📦 Installation

```bash
# Create a new conda environment named "field-nice-gui" with Python 3.10
conda create --name field-nice-gui python=3.10

# Activate the newly created environment
conda activate field-nice-gui

# Clone the project repository from GitHub
# If you're prompted for a username/password, use a GitHub Personal Access Token (PAT) instead.
# You can generate a token at: https://github.com/settings/tokens
# Alternatively, if cloning fails, visit the repository in your browser and download the ZIP archive.
git clone https://github.com/earthroverprogram/field-nice-gui

# Change directory into the cloned project folder
cd field-nice-gui

# Install all required Python packages from the requirements.txt file
pip install -r requirements.txt
```

> ⚠️ **Important Notice**  
> **NiceGUI** requires **NumPy 2**, but **Pyrocko (Snuffler)** is only compatible with **NumPy 1**.  
> These two **cannot be installed in the same environment**.

- If **Snuffler** is already installed on your system, you don't need to reinstall it. The UI will attempt to automatically detect the existing `snuffler` executable (e.g. `/opt/anaconda3/envs/some_env/bin/snuffler`). If auto-detection fails, you can manually specify this path in the UI.

- If **Snuffler** is not yet installed, you **must install Pyrocko** in a **separate conda environment**, e.g.,

  ```bash
  conda create --name snuffler-env python=3.10
  conda activate snuffler-env
  pip install pyrocko
  ```
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
