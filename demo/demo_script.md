

## 0 · Intro (0:00 – 1:30)

On camera.

> "Hi, I'm Harini, and this is Team 16's data mining final project demo. I'm joined by gun madan. We built a system that detects unusual daily weather events — heat waves, cold snaps, heavy rain — in three U.S. cities over the last ten years. Today I'll walk through the problem, our pipeline, the three detectors we compare, and the results.
>
> The project is a Python pipeline built on NOAA Climate Data Online. We'll run it end to end live, so you'll see everything from raw CSV to the anomaly plots we put in the report."

Switch to slide showing: title, team names, course, GitHub link.

---

## 1 · Problem motivation (1:30 – 3:00)

Screen: show `reports/Team_16_FinalReport.pdf` page 1, the abstract.

> "A climate anomaly is any daily observation that deviates strongly from the typical seasonal behavior for that location — not a raw threshold, but a deviation from what's expected on that specific day of the year, in that specific city.
>
> That framing matters because a naive top-2% rule on raw temperature just flags every July day in Chicago and misses real deviations in October. Any useful detector has to model seasonality explicitly."

Scroll to the Introduction section; point at the three-city map text. Mention Chicago = high variability, NYC = moderate, LA = low.

---

## 2 · The dataset (3:00 – 4:30)

Screen: open `data/OriginalWeatherDataset.csv` in VS Code or terminal (`head -3`, `wc -l`).

> "This is the raw pull from NOAA's Climate Data Online portal — 11,190 rows, 50 columns, three stations:
> - Chicago O'Hare (USW00094846)
> - New York Central Park (USW00094728)
> - Los Angeles Downtown USC (USW00093134)
>
> Coverage is January 2016 to April 2026, roughly ten years per city. We keep three variables: TMAX, TMIN, PRCP."

Show `head -1` and point to the columns we keep.

---

## 3 · Repo walkthrough (4:30 – 6:00)

Screen: `tree` or folder view of the GitHub repo.

> "Here's how the repo is organized. `data/` has the raw CSV and the cleaned outputs. `src/` has the reusable modules — `data_cleaning.py`, `feature_engineering.py`, `models.py`, `evaluation.py`, `visualize.py`. `notebooks/` has three notebooks matching our three stages. `reports/` has the proposal, mid-project report, and this final report. `outputs/` has every figure and table the pipeline generates."

Open `src/data_cleaning.py` briefly — show `get_city` and `clean`. Then `feature_engineering.py`, point at the rolling-mean groupby: "this is computed per city, not globally — that bug cost us half a day early on."

---

## 4 · Run the full pipeline live (6:00 – 9:30)

Screen: terminal.

```
cd Team16-WeatherAnomalyDetection
pip install -r requirements.txt   # show, then skip if already installed
python src/run_pipeline.py
```

While it runs, narrate:

> "The pipeline loads the raw CSV, normalizes city names, drops rows with missing TMAX/TMIN/PRCP, engineers day-of-year + rolling statistics, splits time-based at 2023-01-01, fits all three detectors on 2016–2022, then scores 2023–2026 and writes every figure and table."

When it finishes, show the last ~15 lines: confirm rows and output paths.

---

## 5 · EDA (9:30 – 11:00)

Open `notebooks/01_data_cleaning_and_EDA.ipynb` (pre-executed).

Walk through:
1. Boxplot of TMAX by city — "Chicago has the widest range, LA is tight."
2. Monthly average TMAX — "classic seasonal curves, LA roughly flat."
3. Correlation heatmap — "TMAX/TMIN correlated at 0.95, precipitation nearly independent."

---

## 6 · Detection methods (11:00 – 14:00)

Open `notebooks/02_modeling.ipynb`.

### Seasonal z-score (~1 min)

> "For each (city, day-of-year) we compute the mean and standard deviation on the training window only. Then `z = (x - mean) / std`. Flag when `|z| > 2`."

Show the fit/apply cells. Show the resulting `anomaly_z_TMAX` column.

### MAD (~1 min)

> "Same idea but with median and median absolute deviation instead of mean and std. More robust to heavy tails — which matters a lot for precipitation. Flag when the normal-consistent MAD score exceeds 3.5."

Show MAD fit/apply.

### Isolation Forest (~1 min)

> "A tree-based unsupervised detector. Instead of modeling seasonality explicitly it learns the joint density of our engineered features and isolates points that are easy to separate — short tree paths mean anomalous. Trained on `[TMAX, TMIN, PRCP, temp_range, rolling_mean]` with contamination 0.02."

Show the `IsolationForest(...)` line and the resulting `anomaly_if` column.

---

## 7 · Evaluation (14:00 – 17:30)

Open `notebooks/03_evaluation.ipynb`.

### Proxy-label precision (~1 min)

Show the metrics table and the precision bar chart.

> "We don't have labeled anomalies, so we use the extreme 2% of each city-variable distribution as proxy events. Isolation Forest hits 0.62 precision on precipitation — it's very sharp on rain. Z-score has the highest recall on temperature, which makes sense because it's the most sensitive detector."

### Agreement (~1 min)

Show the TMAX Jaccard heatmap.

> "Jaccard agreement is low across the board. Z-score and MAD overlap at about 0.26 — same shape, different thresholds. Isolation Forest only agrees with either at 0.05 — it's finding a fundamentally different set of anomalies because it's working off the joint distribution."

### Stability (~1 min)

Show the stability-over-time plot.

> "Anomaly rates are stable and low, 1–3%, from 2016 to 2022. Then they jump sharply in 2023 and keep climbing. Some of that is the train/test boundary, but every detector and every city shows the jump, so it's not a single-method artifact — it's consistent with how anomalous the last few years have actually been."

### Qualitative validation (~30 sec)

Show the top-anomalies tables.

> "The top flags line up with real events: May 2025 Southern California heatwave, January 2024 Chicago cold wave, the September 2023 NYC flood — 5.48 inches in a day — and the Chicago July 2023 severe thunderstorm."

Show `anom_NYC_TMAX_Zscore.png` and `anom_Chicago_PRCP_MAD.png` as closers.

---

## 8 · Conclusions + future work (17:30 – 19:00)

On camera.

> "Three takeaways. First, no single detector is best everywhere — Isolation Forest wins on precipitation precision, Z-score wins on temperature recall, MAD is the most conservative. Second, detectors disagree strongly, so they're surfacing complementary notions of 'anomaly' — in a real deployment you'd treat them as an ensemble. Third, we see a genuine post-2022 drift in anomaly rates that's stable across methods and cities, which is consistent with recent extreme-weather years.
>
> Things we'd do with more time: seasonal-trend decomposition using STL, an LSTM-based residual model, and proper labeled ground truth from Storm Prediction Center records rather than proxy labels."

---

## 9 · Wrap (19:00 – 20:00)

On camera.

> "Everything you just saw is on GitHub at the link below. The README has step-by-step instructions to reproduce every figure and table. Thanks for watching."

Show the repo URL on screen.

END.

---

### Recording checklist

- [ ] Webcam + face visible at intro and outro (rubric requirement).
- [ ] Clean desktop — no extra tabs or notifications.
- [ ] Run the pipeline once before recording so `pip install` is already satisfied.
- [ ] Rehearse the detector explanations to keep each under 60 seconds.
- [ ] Target 18–20 min total. Trim with cuts if longer.
- [ ] Export 1080p, upload to Google Drive, set sharing to "Anyone with the link", paste the link into the final report's Section VI before submission.
