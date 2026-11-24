"""
test_core.py — regression tests for the logic most likely to break silently.

RUN IT:  python tests/test_core.py
(No pytest needed — plain asserts, so it works with the pinned requirements.)

WHY THESE TESTS AND NOT OTHERS: each one guards a bug that was actually found
during the audit, or an invariant whose failure would be SILENT — producing
confident wrong numbers rather than an error message. Those are the failures
worth automating; everything else shows up the moment you open the dashboard.
"""

import json
import os
import sys

import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from rul_triage import (  # noqa: E402
    estimate_rul_days,
    apply_triage_weighting,
    build_priority_alert,
    FEATURE_DISPLAY_NAMES,
    RISK_CRITICAL_THRESHOLD,
    RISK_WARNING_THRESHOLD,
)
from llm_dispatch import generate_sop_offline_fallback  # noqa: E402

PASSED, FAILED = [], []


def check(name, fn):
    try:
        fn()
        PASSED.append(name)
        print(f"  PASS  {name}")
    except AssertionError as exc:
        FAILED.append((name, str(exc)))
        print(f"  FAIL  {name}\n        {exc}")


# ---------------------------------------------------------------------------
# RUL
# ---------------------------------------------------------------------------

def test_rul_is_monotone():
    """A healthier device must always get MORE days than a sicker one."""
    assert estimate_rul_days(0.1) > estimate_rul_days(0.5) > estimate_rul_days(0.9)


def test_rul_clamps_both_ends():
    """p=0 must not promise 'never fails'; p=1 must not report 0 days."""
    assert estimate_rul_days(0.0) == 361.4, estimate_rul_days(0.0)
    assert estimate_rul_days(1.0) >= 1.0
    assert estimate_rul_days(-5.0) == estimate_rul_days(0.0)


def test_rul_survives_garbage_input():
    """A bad value must not crash the dashboard — it defaults to 0.5."""
    assert estimate_rul_days("not a number") == estimate_rul_days(0.5)
    assert estimate_rul_days(None) == estimate_rul_days(0.5)
    assert estimate_rul_days(float("nan")) == estimate_rul_days(0.5)


def test_rul_respects_custom_max_days():
    assert estimate_rul_days(0.5, max_days=180) == 90.0


# ---------------------------------------------------------------------------
# Clinical triage — the core innovation
# ---------------------------------------------------------------------------

def test_icu_outranks_storage_at_lower_raw_risk():
    """THE headline claim: a 60% ICU device must outrank an 80% storage device."""
    w = {"ICU": 1.5, "General Ward": 1.0, "Storage": 0.7}
    icu = apply_triage_weighting(0.60, "ICU", w)
    storage = apply_triage_weighting(0.80, "Storage", w)
    assert icu == 90.0, icu
    assert storage == 56.0, storage
    assert icu > storage


def test_unknown_ward_is_neutral_not_punitive():
    """An unrecognised ward must neither inflate nor deflate the score."""
    assert apply_triage_weighting(0.5, "Nowhere", {"ICU": 1.5}) == 50.0


def test_triage_survives_garbage_input():
    assert apply_triage_weighting("bad", "ICU", {"ICU": 1.5}) == 75.0


# ---------------------------------------------------------------------------
# Alert assembly
# ---------------------------------------------------------------------------

def test_alert_translates_driver_names():
    """Raw column names must never reach the user or the LLM prompt.
    This is the regression test for audit bug #3."""
    alert = build_priority_alert(
        "Pump", "Cardiovascular Devices", 0.85, "ICU",
        {"ICU": 1.5},
        [{"feature": "manufacturer_risk_tier_encoded", "impact": 0.3},
         {"feature": "age_fraction", "impact": 1.2}],
    )
    names = [d["feature"] for d in alert["top_drivers"]]
    assert names == ["Manufacturer Recall History", "Device Age"], names


def test_alert_action_tiers():
    w = {"ICU": 1.0}
    hi = build_priority_alert("d", "c", 0.90, "ICU", w, [])["recommended_action"]
    md = build_priority_alert("d", "c", 0.50, "ICU", w, [])["recommended_action"]
    lo = build_priority_alert("d", "c", 0.10, "ICU", w, [])["recommended_action"]
    assert "immediate" in hi.lower(), hi
    assert "maintenance window" in md.lower(), md
    assert "routine" in lo.lower(), lo


def test_thresholds_are_probabilities_not_percentages():
    """Guards the unit confusion that existed when 70/40 and 0.7/0.4 were
    duplicated across three files."""
    assert 0 < RISK_WARNING_THRESHOLD < RISK_CRITICAL_THRESHOLD < 1


# ---------------------------------------------------------------------------
# Feature contract — the silent-failure guards
# ---------------------------------------------------------------------------

def test_every_model_feature_has_a_display_name():
    """If a feature is added to the model but not to FEATURE_DISPLAY_NAMES,
    users see a raw column name. Catch it here instead."""
    cols = joblib.load(os.path.join(BASE_DIR, "models", "feature_columns_realistic.pkl"))
    missing = [c for c in cols if c not in FEATURE_DISPLAY_NAMES]
    assert not missing, f"features with no display name: {missing}"


def test_both_variants_share_the_same_feature_contract():
    a = joblib.load(os.path.join(BASE_DIR, "models", "feature_columns_realistic.pkl"))
    b = joblib.load(os.path.join(BASE_DIR, "models", "feature_columns_optimized.pkl"))
    assert a == b, "variant feature orders diverged"


def test_schema_categories_match_the_trained_encoder():
    """THE most valuable test here. If device_schema.json and the encoder ever
    drift apart, every category is encoded to the wrong integer and the model
    returns confident, plausible, completely wrong predictions — with no error
    raised anywhere."""
    with open(os.path.join(BASE_DIR, "config", "device_schema.json")) as f:
        schema = json.load(f)
    schema.pop("_comment", None)
    enc = joblib.load(os.path.join(BASE_DIR, "models", "category_encoder_realistic.pkl"))
    assert set(schema.keys()) == set(enc.classes_), (
        f"only in schema: {set(schema) - set(enc.classes_)} | "
        f"only in encoder: {set(enc.classes_) - set(schema)}"
    )


def test_every_schema_category_declares_ranges_for_its_fields():
    with open(os.path.join(BASE_DIR, "config", "device_schema.json")) as f:
        schema = json.load(f)
    schema.pop("_comment", None)
    for cat, spec in schema.items():
        for field in spec["fields"]:
            assert field in spec["ranges"], f"{cat} lists '{field}' with no range"
            lo, hi = spec["ranges"][field]
            assert lo < hi, f"{cat}.{field} has an inverted range"
        assert spec["ward_criticality"] in ("ICU", "General Ward", "Storage")


def test_models_load_without_version_warnings():
    """requirements.txt is pinned to the versions that wrote these pickles."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for v in ("realistic", "optimized"):
            joblib.load(os.path.join(BASE_DIR, "models", f"classifier_{v}.pkl"))
            joblib.load(os.path.join(BASE_DIR, "models", f"category_encoder_{v}.pkl"))


# ---------------------------------------------------------------------------
# Offline SOP — the resilience guarantee
# ---------------------------------------------------------------------------

def test_offline_sop_is_substantial_and_labelled():
    sop = generate_sop_offline_fallback(
        "Infusion Pump", "Cardiovascular Devices", 92.0,
        [{"feature": "Device Age", "impact": 1.5}],
    )
    assert len(sop) > 800, f"offline SOP too short: {len(sop)} chars"
    assert "URGENCY LEVEL: HIGH" in sop
    assert "Offline Mode" in sop, "offline output must be labelled as such"
    assert "end of its expected service life" in sop, "driver guidance missing"


def test_offline_sop_covers_missing_flag_drivers():
    """The four *_was_missing display names must have real guidance, not the
    generic 'investigate further' fallback."""
    sop = generate_sop_offline_fallback(
        "Scanner", "Orthopedic Devices", 55.0,
        [{"feature": "Vibration Sensor Not Tracked", "impact": 0.4}],
    )
    assert "no vibration sensor" in sop.lower(), sop[:400]


def test_offline_sop_urgency_matches_risk():
    for pct, word in [(95.0, "HIGH"), (55.0, "MEDIUM"), (10.0, "LOW")]:
        sop = generate_sop_offline_fallback("d", "Dental Devices", pct, [])
        assert f"URGENCY LEVEL: {word}" in sop, f"{pct}% -> expected {word}"


if __name__ == "__main__":
    print("Running MedEQ core tests...\n")
    for name, fn in sorted(
        (n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)
    ):
        check(name, fn)

    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    sys.exit(1 if FAILED else 0)
