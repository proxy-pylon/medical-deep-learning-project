Here’s a clean, drop-in `README.md` you can use. It stays in-bounds: Linux setup, venv or Docker, installing `requirements.txt`, and pulling data from Kaggle. No phantom Makefiles or other imaginary friends.

```markdown
# Final Project

Reproducible setup for training and evaluation. This document covers:

- Creating a Python virtual environment **or** using Docker on Linux
- Installing dependencies from `requirements.txt`
- Downloading data from Kaggle using the Kaggle API
- Project layout and expected data paths

---

## Project layout

```

final-project/
├─ data/            # put raw/processed datasets here
├─ output/          # training logs, checkpoints, predictions
├─ report/          # notebooks, figures, write-ups
├─ check_gpu.py
├─ model.py
├─ requirements.txt
└─ README.md

````

> If you’re on WSL, the project path may look like `\\wsl.localhost\Ubuntu-24.04\home\<user>\...`.

---

## Option A: Python virtual environment (recommended)

### 1) Prerequisites
- Python 3.10+ installed (`python3 --version`)
- `pip` and `venv` modules available

On Ubuntu/Debian:
```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip
````

### 2) Create and activate the venv

From the project root:

```bash
python3 -m venv .venv
# Activate for current shell
source .venv/bin/activate
# Upgrade packaging tools inside the venv
python -m pip install --upgrade pip setuptools wheel
```

> Each new terminal session: run `source .venv/bin/activate` before working.

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

---

## Option B: Docker (no local Python needed)

### 1) Install Docker

Follow your distro instructions. On Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo $VERSION_CODENAME) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io
```

Add your user to the `docker` group (optional):

```bash
sudo usermod -aG docker "$USER"
# Log out and back in for this to take effect
```

### 2) Run a container and install deps inside it

From the project root:

```bash
docker run --rm -it \
  -v "$PWD":/work \
  -w /work \
  python:3.11-slim bash
```

Inside the container shell:

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

> To use a GPU, launch an image that includes CUDA and run with `--gpus all` if your host supports it.

---

## Downloading data from Kaggle

We use the Kaggle API to fetch datasets directly into `data/`.

### 1) Install the Kaggle CLI

If you’re using the venv:

```bash
pip install kaggle
```

In Docker: run the same command inside the container.

### 2) Set up your Kaggle credentials

1. Go to [https://www.kaggle.com](https://www.kaggle.com) → your profile → **Account** → **API** → **Create New Token**.
   This downloads a file named `kaggle.json`.
2. Place it at `~/.kaggle/kaggle.json` and set permissions:

   ```bash
   mkdir -p ~/.kaggle
   mv /path/to/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

> On WSL, `~` refers to your Linux home (not Windows). Use your Linux terminal.

### 3) Download a dataset

Replace the placeholder with the actual dataset slug you’re using.

* **Public dataset example** (dataset page shows the slug):

  ```bash
  kaggle datasets download -d <owner>/<dataset-slug> -p data/ --unzip
  ```

  Example:

  ```bash
  kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p data/ --unzip
  ```

* **Competition data** (requires competition rules acceptance in the browser first):

  ```bash
  kaggle competitions download -c <competition-name> -p data/
  # Unzip if needed
  unzip -q data/*.zip -d data/
  ```

### 4) Verify files

After download and unzip, you should have files under `data/`. Adjust any paths in your scripts if needed.

---

## Quick checks

* Verify Python sees the packages:

  ```bash
  python -c "import torch, pandas, sklearn; print('ok')"
  ```
* Optional GPU check:

  ```bash
  python check_gpu.py
  ```

---

## Notes

* Always activate the venv (or enter the Docker container) before running any training or evaluation scripts.
* If `pip install -r requirements.txt` fails due to system packages, install build tools:

  ```bash
  sudo apt-get update
  sudo apt-get install -y build-essential
  ```
* If a specific library needs system headers (e.g., OpenCV, libGL), install them:

  ```bash
  sudo apt-get install -y libgl1 libglib2.0-0
  ```

---

## Reproducibility

Use fixed seeds in your scripts where applicable and record exact package versions in `requirements.txt`. When updating dependencies, consider pinning versions to avoid surprises.

```

Enjoy responsibly.
```
