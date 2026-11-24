# MedEQ — Start Here

Medical Equipment Failure Prediction Agent.
Everything is pre-trained and included. You do **not** need to download a dataset
or train anything to run the demo.

---

## Fastest path (Windows, ~5 minutes)

1. Unzip the folder anywhere (avoid paths with unusual characters).
2. Open the folder in VS Code → **Terminal → New Terminal**.
3. Run these, in order:

```
setup.bat
```

Wait for it to finish, then open **two** terminals:

```
Terminal 1:   run_api.bat
Terminal 2:   run_dashboard.bat
```

4. The dashboard opens at **http://localhost:8501**
   The API docs are at **http://127.0.0.1:8000/docs**

## macOS / Linux

```bash
./setup.sh
./run_api.sh          # terminal 1
./run_dashboard.sh    # terminal 2
```

## If you prefer to do it manually

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux

pip install -r requirements.txt

uvicorn src.api:app --reload --port 8000     # terminal 1
streamlit run app_streamlit.py               # terminal 2
```

---

## The LLM key (optional)

The app runs fully without it. Without a key, the **Generate SOP Work Order**
button produces a detailed offline template instead of an LLM-written one —
that is the built-in resilience behaviour, not a failure.

To enable the LLM:

1. Get a free key at https://aistudio.google.com/apikey
2. Copy `.env.example` to `.env`
3. Paste your key into it

**Never put the real `.env` in a zip you share, and never commit it.**
If a key has been shared, revoke it in AI Studio and create a new one.

---

## What you should see

| Tab | What it does |
|---|---|
| **Device Risk List** | Scans 50 devices, ranks them by clinical triage priority, shows SHAP drivers, generates an SOP work order for any device |
| **What-If Simulator** | Move the sliders and watch risk, RUL and SHAP drivers update live |
| **Model Comparison** | The 95% and 99% models side by side, with the honest explanation of the difference |
| **RUL Estimator** | Turns a failure probability into a days-remaining countdown |

Click **Run Risk Scan** on tab 1 first — the other tabs work independently.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| "Backend API is not running" | The API terminal isn't started | Run `run_api.bat` first and leave it open |
| Port 8000 already in use | An old API is still running | Close the other terminal, or use `--port 8001` and set `MEDEQ_API_URL` |
| "LLM Offline" pill | No key, or the daily free quota is used up | Expected. The offline SOP template still works. |
| `ModuleNotFoundError` | Dependencies not installed, or the wrong Python is selected in VS Code | Re-run `setup.bat`; in VS Code press Ctrl+Shift+P → "Python: Select Interpreter" → pick `.venv` |
| SOP generation is slow | The LLM is rotating through its fallback model chain | Normal, wait up to ~90 seconds |

---

## Rebuilding things from scratch (optional)

Nothing below is needed for the demo — everything is already built.

```bash
# Regenerate both synthetic training datasets (overwrites data/)
python src/synthetic_telemetry.py --force

# Retrain both models on that data (takes several minutes)
python src/train_model.py

# Re-run the imputation experiment; writes outputs/imputation_experiment.csv
python src/experiment_imputation.py

# Rebuild the real-data features. Needs the original Kaggle CSVs:
# put them in a "dataset/" folder here, or set MEDEQ_DATASET_DIR.
python src/real_data_features.py
```

If you regenerate the data, retrain afterwards so the saved models match it.

See `README.md` for the full explanation of what each file does and why.
