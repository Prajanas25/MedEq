# Medical Equipment Failure Prediction Agent (MedEQ)

Predicts which hospital devices are heading for failure, explains **why** using
SHAP, ranks them by **clinical consequence** rather than raw risk, and generates
a maintenance **SOP work order** via an LLM (with an offline fallback).

**To run it, see [START_HERE.md](START_HERE.md).** This file explains what
everything is and why it was built that way.

---

## Project structure

```
medeq_project/
├── START_HERE.md                # Quick start - read this first
├── setup.bat / setup.sh         # One-time environment setup
├── run_api.bat / run_api.sh     # Starts the backend
├── run_dashboard.bat / .sh      # Starts the dashboard
├── .env.example                 # Template for the optional Gemini API key
│
├── config/
│   ├── device_schema.json       # REUSABILITY LAYER: 16 device categories,
│   │                            #   which telemetry fields each one has,
│   │                            #   realistic ranges, and ward criticality
│   └── triage_weights.json      # Ward multipliers (ICU 1.5 / General 1.0 / Storage 0.7)
│
├── data/
│   ├── real_devices_sample.csv          # REAL: 35,597 devices from Kaggle devices.csv
│   ├── manufacturer_recall_stats.csv    # REAL: 31,827 manufacturers, real recall counts
│   ├── category_recall_stats.csv        # REAL: recall rate per category (all 1.0 - see below)
│   ├── synthetic_telemetry_realistic.csv   # GENERATED: trains the 95% model
│   └── synthetic_telemetry_optimized.csv   # GENERATED: trains the 99% model
│
├── models/                      # Pre-trained. Nothing to run before the demo.
│   ├── classifier_realistic.pkl / classifier_optimized.pkl
│   ├── regressor_realistic.pkl  / regressor_optimized.pkl
│   ├── category_encoder_*.pkl   / feature_columns_*.pkl
│   └── metrics.json             # Accuracy / precision / recall / F1 for both
│
├── src/
│   ├── real_data_features.py    # Joins all 3 Kaggle files -> real risk features
│   ├── synthetic_telemetry.py   # Generates both synthetic training datasets
│   ├── train_model.py           # Trains + tunes + evaluates + saves both variants
│   ├── experiment_imputation.py # Standalone experiment: which fill strategy is best
│   ├── rul_triage.py            # RUL countdown + clinical triage prioritisation
│   ├── llm_dispatch.py          # SHAP-to-SOP LLM agent (Gemini) + offline fallback
│   └── api.py                   # FastAPI: /predict /explain /dispatch /model_metrics
│
├── tests/
│   └── test_core.py             # 18 regression tests: python tests/test_core.py
│
├── app_streamlit.py             # 4-tab dashboard
└── requirements.txt             # Pinned - see the note inside the file
```

---

## The core problem, and how it was solved

The supplied Kaggle dataset (*faulty-medical-devices-global-dataset*) turned out
to be a **recall registry, not a monitoring feed**. Three structural facts made
the obvious approach impossible:

1. **Every device in it already has a recall event** — there are zero "healthy
   device" examples to learn against. The proof is in
   `data/category_recall_stats.csv`: the recall rate is exactly `1.0` for all
   16 categories.
2. **There are no sensor readings at all** — no temperature, vibration, voltage
   or usage hours anywhere in the dataset.
3. **There is no time series** — so there is no "days until failure" ground truth.

The brief explicitly anticipated this: use case #8 states that *"synthetic
medical device telemetry data can also be created."*

So the solution uses a **hybrid data strategy**:

| Layer | What it provides | Real or synthetic |
|---|---|---|
| Device identity | `device_id`, `device_name`, `classification`, `manufacturer_id` | **Real** |
| Manufacturer risk | `manufacturer_risk_tier` (low/medium/high), from counting real recall events per manufacturer | **Real** |
| Sensor readings | `temperature`, `vibration`, `voltage`, `hours_used` | Synthetic |
| Device age | `age_fraction` | Synthetic |
| Failure label | `failed`, `true_failure_probability` | Synthetic |

The synthetic part is not random noise. Sensor values follow an **accelerating
degradation curve** (`base + (max-base) x age^2 + noise`), and failure
probability blends age, sensor drift, and the **real** manufacturer recall tier:

```
p = 0.45 x age + 0.45 x sensor_closeness + tier_boost      (tier_boost: 0 / 0.08 / 0.15)
```

---

## How all three real dataset files are genuinely used

- **`devices.csv`** → real device names, the 16-category taxonomy, manufacturer links
- **`manufacturers.csv`** → real manufacturer identities
- **`events.csv`** → real recall/safety events, counted per manufacturer

Joining `events.csv` to `manufacturers.csv` produces a real recall count per
manufacturer, binned into low/medium/high tiers. That tier is a **real input
feature** to the failure model, and it demonstrably moves predictions: holding
every other input constant and changing only the tier from `high` to `low`
shifts predicted risk from **99.24% to 98.66%**.

### The honest finding — do not skip this when presenting

We first tried a classifier predicting recall `type` (Recall / Field Safety
Notice / Safety alert) from real data alone. It scored **99.99%**.

We investigated it instead of reporting it, and found that **every country in
this dataset files exactly one `type` value, always** — the USA and Canada only
ever file "Recall", Germany and Switzerland only ever file "Field Safety
Notice". `type` is a **regulatory naming convention per country**, not a
medical outcome. The model was memorising a naming rule.

That model was therefore **dropped as a showcase result**. In the code
(`src/real_data_features.py`) it now:

- does **not** run by default (pass `--train-type-model` if you want it),
- has the leaking `country` feature **removed** from its feature list, so any
  score it reports is genuine, and
- ships no saved artefacts, so there is no stale 99.99% metrics file in `models/`.

What replaced it is the signal that *is* genuinely predictive: manufacturer
recall history.

---

## Two model variants, shown transparently

`train_model.py` trains and saves **both**:

| Variant | Accuracy | Precision | Recall | F1 | What it means |
|---|---|---|---|---|---|
| **realistic** | **95.03%** | 91.93% | 86.85% | **0.8932** | 15% of labels carry deliberate randomness, so genuine irreducible error exists. **Recommended.** |
| **optimized** | **99.20%** | 97.61% | 98.74% | **0.9817** | Label is exactly `probability > 0.5`, a direct function of the features. Measures the *label design*, not predictive difficulty. |

Both are selectable through the API's `model_variant` field, and the
dashboard's **Model Comparison** tab shows them side by side with the reasoning
written into the UI.

This turns the most predictable question — *"why is your accuracy so high?"* —
into a demonstrated strength: the tradeoff was tested, measured, and is
explained rather than hidden behind one number.

---

## Class imbalance

Roughly 24% of synthetic rows are failures. Without correction, XGBoost drifts
toward always predicting "safe", which scores 76% accuracy while catching zero
failures. `scale_pos_weight = n_negative / n_positive` (3.18 for the realistic
model) makes each missed failure cost ~3.18x as much during training.

Measured effect: recall on the failed class was **0.29** without it, and
**0.87** in the final tuned model. In a patient-safety context a missed failure
is far worse than a false alarm, so this is the correct tradeoff.

---

## Missing data strategy

Missingness here is **structural**, not random: Orthopedic Devices have no
temperature field in `device_schema.json`, so *every* orthopedic row lacks
temperature. Approach used:

- add a `<field>_was_missing` flag column, **then** fill the value with 0
- the flag order matters — computing it after the fill would produce four
  all-zero columns and silently destroy the mechanism

`src/experiment_imputation.py` tested this against global-mean and
category-mean imputation and found zero-fill **tied for best**, because when a
whole category lacks a field there is nothing real to average. The script now
writes its results to `outputs/imputation_experiment.csv`.

---

## What each piece demonstrates (mapped to the rubric)

- **Model performance metrics** — `models/metrics.json`, plus train-set metrics
  and a persisted confusion matrix on retrain
- **Explainability** — TreeSHAP in `/predict` and `/explain`, with human-readable
  driver names and log-odds units labelled
- **Innovation** — clinical triage prioritisation, SHAP-to-SOP agentic dispatch,
  RUL countdown, What-If simulator
- **Reusability** — add a device category by editing `config/device_schema.json`
  only; the generator, the API's missing-flag logic, and the simulator's sliders
  all adapt with no code change
- **Resilience** — `/dispatch` degrades LLM → model fallback chain → offline
  template and never returns an error; the dashboard states plainly when the
  backend or LLM is unavailable rather than showing a false "all clear"

---

## Tests

```bash
python tests/test_core.py
```

18 plain-assert tests (no pytest needed). They cover the failures that would
otherwise be **silent** rather than loud - notably that `device_schema.json`
and the trained category encoder still agree. If those two ever drift apart,
every category is encoded to the wrong integer and the model returns
confident, plausible, completely wrong predictions with no error anywhere.

---

## Known limitations (be ready to explain these)

1. **Every performance number describes synthetic data we designed.** The
   pipeline, explanations, triage logic and resilience transfer to real
   telemetry; the accuracy figure does not.
2. **RUL is a transparent formula, not a validated prediction.**
   `days = 365 x (1 - probability)`. A trained "days until failure" model would
   need real failure timestamps, which this dataset does not contain.
3. **Triage weights are policy, not evidence.** They live in a JSON file so a
   hospital administrator can change them without a developer or a retrain.
4. **No device heartbeat / OFFLINE state is implemented.** The intended design
   is that a device with no telemetry for a period is marked
   "OFFLINE — status unknown" rather than showing a stale "safe". That is a
   described design, not shipped code.
5. **MCP is not implemented.** The three endpoints are plain REST.
6. **The country encoder covers 35 values**, not 39 — the encoder is fitted
   after filtering to the three main recall types and dropping rows with a null
   classification or country.
