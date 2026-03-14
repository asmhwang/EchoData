# EchoData — Synthetic Data for Data protectoin

A cross-platform web GUI for generating privacy-preserving synthetic data and comparing it against the original using machine learning.

Works on **macOS**, **Windows**, and **Linux** — runs as a local Flask server and opens in your browser.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run (opens browser automatically)
python app.py
```

Then open **http://localhost:5111** if it doesn't open automatically.

## How It Works

1. **Load** — Drag-and-drop any CSV file. Auto-detects numeric and categorical columns.
2. **Configure** — Adjust the privacy budget (ε) and number of synthetic rows.
3. **Generate** — The ShatteredSynth engine decomposes your data into noisy statistical fragments and reassembles synthetic rows.
4. **Compare** — Train Random Forest models on both datasets to verify predictive patterns are preserved.
