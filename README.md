
# Stock Market Education App – Content Processing Pipeline

This repository contains a **pipeline for scraping, processing, and summarizing educational content** for a beginner-friendly stock market education app. It uses **web crawling, LLM summarization, and structured YAML generation** to convert raw text into ready-to-use study materials.

It now also includes a separate **ML trading simulation module** for an FYP-style demo where:

- a stock movement classifier predicts next-day direction
- multiple AI trader personalities convert that prediction into `BUY` / `SELL` / `HOLD`
- a backtest compares which personality performs best over time


## Project Structure

```
FYP-Data/
│
├─ data/
│  ├─ raw/                # Raw scraped text files
│  │  ├─ <main_topic>/
│  │  │  ├─ <subtopic>/
│  │  │  │  ├─ file1.txt
│  │  │  │  ├─ file2.txt
│  │  │  │  └─ ...
│  │  │  └─ ...
│  │  └─ ...
│  └─ processed/          # Generated YAML summaries (output)
│     ├─ <main_topic>/
│     │  ├─ <subtopic>/
│     │  │  ├─ file1_summary.yaml
│     │  │  └─ ...
│     │  └─ ...
│     └─ ...
│
├─ scripts/
│  └─ process.py          # Main LLM pipeline script
├─ trading/
│  ├─ __init__.py
│  └─ run_simulation.py   # ML + multi-agent trading demo
│
├─ models/
│  └─ Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
│
└─ README.md

````

---

## Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/FYP-Data.git
cd FYP-Data
```

2. Prepare directories:

```bash
mkdir -p data/raw
mkdir -p data/processed
```

3. Create and activate virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
deactivate
```

Make sure to have python and venv installed in ubuntu bash: Python version used: 3.12.3
```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip -y
```

4. Install project dependencies:
```bash
pip install -r requirements.txt
```
5. Download your model and place the `.gguf` file in the root or `models/` directory.

6.  Inside `.env`, adjust:

        | Variable           | 
        | ------------------ |
        | `GOOGLE_API_KEY`   |
        | `GOOGLE_CSE_ID`    |
        | `RAW_DIR`          |
        | `PROCESSED_DIR`    |
        | `MODEL_PATH`       |

7. Create Custom Search API:

    Go to Google Cloud Console
    Create a projecT
    APIs & Services → Library
    Enable Custom Search API
    APIs & Services → Credentials → Create API key

8. Create a Custom Search Engine (CSE)

    Go to https://programmablesearchengine.google.com
    Create new search engine

    Important setting:
    Sites to search → enter www.google.com
    Then enable “Search the entire web”
    Copy your Search Engine ID (CX)

9. Run code:
```bash
Webscraping: python main.py
Processing:  python process.py
Trading demo: python -m trading.run_simulation --ticker AAPL --plot
```


---


## 📝 Pipeline Overview

The pipeline performs the following steps:

### 1. Raw text reading

* Reads all `.txt` files under `data/raw/<main_topic>/<subtopic>/`.
* Supports **any number of topics/subtopics**.
* Can safely truncate long text chunks to avoid exceeding the LLM context.

### 2. Micro-summary generation

* Uses an **LLM (Meta-LLaMA)** to generate a concise summary of each text chunk.
* Removes fluff and focuses on **definitions, key points, and examples**.
* Produces one micro-summary per chunk.

### 3. Final structured summary

* Uses the LLM to generate a **full lesson YAML** for the educational app.

* Includes:

  * Title
  * 7–10 sentence summary
  * Key points (3–7 bullets)
  * Examples (1–3)
  * Definitions (simple explanations)
  * Common mistakes (1–2)
  * Questions to think about (1–2)
  * Source citation

* YAML output is stored in `data/processed/<main_topic>/<subtopic>/`.

### 4. One-to-one chunk mapping

* Each `.txt` chunk produces **one corresponding YAML file**.
* Large files can be split into multiple chunks for safety.

---

## Trading Simulation Module

The trading workflow is intentionally separated from the content pipeline so you can present it as a second major FYP component without mixing concerns.

### What it does

`trading/run_simulation.py` runs this flow:

1. Download historical OHLCV data from Yahoo Finance with `yfinance`
2. Build trading features such as returns, moving-average gaps, volatility, and volume change
3. Train a `RandomForestClassifier` to predict whether tomorrow's close will be higher than today's
4. Generate confidence scores with `predict_proba`
5. Let three trader personalities act on those predictions:
   - `conservative`: trades only when confidence is at least `0.70`
   - `balanced`: trades only when confidence is at least `0.60`
   - `aggressive`: always acts on the model prediction
6. Simulate a simple one-share-at-a-time portfolio and save the results

### Run it

```bash
python -m trading.run_simulation \
  --ticker AAPL \
  --start 2019-01-01 \
  --end 2024-01-01 \
  --initial-cash 10000 \
  --plot
```

### Outputs

The script writes these files to `trading/output/` by default:

- `conservative_trades.csv`
- `balanced_trades.csv`
- `aggressive_trades.csv`
- `summary.csv`
- `portfolio_values.png`

### Why this fits the FYP

This gives you a clean architecture you can explain in the report:

- data collection
- feature engineering
- ML prediction
- agent personality strategy
- trading simulation
- performance comparison
