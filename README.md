# Team 16 — Seasonality-Aware Weather Anomaly Detection

CSC 4740/6740 Data Mining — Spring 2026 — Gunn Madan, Harini Mohan

Detecting unusual daily weather events (heat waves, cold snaps, heavy-rain days) for Chicago, New York City, and Los Angeles using ten years of NOAA Climate Data Online observations. We compare three unsupervised detectors on the same time-based train/test split:

1. Seasonal z-score (per city, per day-of-year)
2. Robust MAD (median absolute deviation)
3. Isolation Forest trained on engineered features

See `reports/Team_16_FinalReport.pdf` for the full write-up and numbers.

---

## Required software

| Tool | Version | Notes |
|------|---------|-------|
| Python | 3.10 or 3.11 | Tested on 3.10 |
| pip | 22+ | or conda, either works |
| OS | Linux / macOS / Windows | No OS-specific code |

Python package versions are pinned in `requirements.txt`:

```
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.4.2
matplotlib==3.8.4
seaborn==0.13.2
jupyter==1.0.0
nbconvert==7.16.4
```

Disk: ~10 MB for data, ~5 MB for the outputs the pipeline generates.
RAM: under 1 GB at peak.

---

## Step-by-step: reproduce everything

```bash
# 1. Clone the repo
git clone https://github.com/<your-handle>/Team16-WeatherAnomalyDetection.git
cd Team16-WeatherAnomalyDetection

# 2. Set up a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full pipeline end-to-end
python src/run_pipeline.py
```

Expected output at the end:

```
Loading raw dataset from .../data/OriginalWeatherDataset.csv
Raw rows: 11190
Clean rows: 11188
Saved clean features to .../data/clean_weather_data.csv
Train rows: 7653 (<2023-01-01)
Test rows:  3517  (>=2023-01-01)
Saved model outputs to .../data/model_output_data.csv
Saved full scored frame to .../data/model_output_full.csv
Saved metrics to .../outputs/tables/metrics_by_city_variable_method.csv
Saved summary to .../outputs/tables/metrics_summary.csv
Done.
```

After running, you will find:

- `data/clean_weather_data.csv` — cleaned dataset with engineered features
- `data/model_output_data.csv` — test-window rows with all detector flags and scores
- `data/model_output_full.csv` — same scoring applied across the full 2016–2026 window (used for stability plots)
- `outputs/tables/` — metrics, agreement matrices, stability tables, top-anomaly tables
- `outputs/figures/` — 37 PNG figures (EDA, anomaly time series per city/variable/method, evaluation plots)

Pipeline runtime: ~10–15 seconds on a laptop.

### CLI flags

```
python src/run_pipeline.py \
    --raw data/OriginalWeatherDataset.csv \
    --data-dir data \
    --fig-dir outputs/figures \
    --table-dir outputs/tables \
    --split-date 2023-01-01
```

All flags are optional; the defaults match the layout above.

---

## Running the notebooks

The three notebooks in `notebooks/` mirror the pipeline stages and can be run top-to-bottom in order. They import from `src/` so they stay in sync with the modules.

```bash
jupyter notebook notebooks/
```

1. `01_data_cleaning_and_EDA.ipynb` — load, clean, feature-engineer, EDA plots.
2. `02_modeling.ipynb` — fit Z-score, MAD, Isolation Forest on the training window and score the test window.
3. `03_evaluation.ipynb` — precision / recall / F1 against proxy labels, Jaccard agreement, stability, top anomalies.

To execute them headlessly from the command line:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

---

## Repository layout

```
Team16-WeatherAnomalyDetection/
├── README.md
├── requirements.txt
├── data/
│   ├── OriginalWeatherDataset.csv           # raw NOAA pull
│   ├── clean_weather_data.csv               # generated
│   ├── model_output_data.csv                # generated
│   └── model_output_full.csv                # generated
├── src/
│   ├── data_cleaning.py                     # city normalization, filtering
│   ├── feature_engineering.py               # rolling stats, seasonal keys, time split
│   ├── models.py                            # z-score, MAD, Isolation Forest
│   ├── evaluation.py                        # proxy labels, Jaccard, stability
│   ├── visualize.py                         # matplotlib/seaborn plots
│   └── run_pipeline.py                      # end-to-end driver
├── notebooks/
│   ├── 01_data_cleaning_and_EDA.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_evaluation.ipynb
├── reports/
│   ├── Team_16_Proposal.pdf                 # Phase 1
│   ├── Team_16_MidProjectReport.pdf         # Phase 2
│   ├── Team_16_FinalReport.pdf              # Phase 3, IEEE format
│   └── Team_16_FinalReport.docx             # editable source
├── outputs/
│   ├── figures/                             # generated PNGs
│   └── tables/                              # generated CSVs
└── demo/
    └── demo_script.md                       # ~20 min demo walkthrough
```

---

## Data source

NOAA National Centers for Environmental Information, Climate Data Online (CDO): https://www.ncei.noaa.gov/cdo-web/

Three GHCN stations were pulled for 2016-01-01 through 2026-04-23:

- Chicago O'Hare International Airport — USW00094846
- New York Central Park — USW00094728
- Los Angeles Downtown USC — USW00093134

The raw CSV (`data/OriginalWeatherDataset.csv`) is checked into the repo so the pipeline is fully self-contained. To re-download from NOAA, use the CDO web interface and request daily summaries for the three station IDs above.

---

## Team

- Gunn Madan
- Harini Mohan

Course: CSC 4740/6740 Data Mining, Georgia State University, Spring 2026.
