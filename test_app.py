"""Automated tests: validation rules, boundary predictions, consistency with
training pipeline, and an end-to-end Streamlit run (AppTest)."""
import numpy as np
import pandas as pd
import joblib
from streamlit.testing.v1 import AppTest
from validation import validate_inputs
from predictor import predict_category, load_bundle
from common import load_data, FEATURES_A

TOTAL_MSG = "Total durasi aktivitas yang kamu masukkan melebihi 24 jam. Silakan periksa kembali jawabanmu."
ok = lambda s, t, p, so: {"study": s, "sleep": t, "physical": p, "social": so}


def test_validation():
    assert validate_inputs(ok(5, 5, 0, 0)) == []                 # min
    assert validate_inputs(ok(10, 10, 13, 6)) == [TOTAL_MSG]    # each field at max, total 39 > 24
    assert validate_inputs(ok(7, 7, 5, 4)) == []                 # total 23 ok
    assert validate_inputs(ok(-1, 7, 2, 2)) != []                # negative
    assert validate_inputs(ok(4.9, 7, 2, 2)) != []               # below range
    assert validate_inputs(ok(10.1, 7, 2, 2)) != []              # above range
    assert validate_inputs(ok(7, 4, 2, 2)) != []                 # sleep below range
    assert validate_inputs(ok(7, 11, 2, 2)) != []                # sleep above range
    assert validate_inputs(ok(7, 7, -0.5, 2)) != []              # physical negative
    assert validate_inputs(ok(7, 7, 13.5, 2)) != []              # physical above
    assert validate_inputs(ok(7, 7, 2, 6.5)) != []               # social above
    assert validate_inputs(ok(7, 7, 2, -1)) != []                # social negative
    assert validate_inputs(ok(8, 8, 8, 1)) == [TOTAL_MSG]        # total 25 > 24
    assert validate_inputs(ok(None, 7, 2, 2)) != []              # empty
    assert validate_inputs({"study": None, "sleep": None, "physical": None, "social": None}) != []
    print("validation: OK")


def test_thresholds_follow_model():
    # Boundary values, compared with the saved model's own predictions (no manual rules)
    bundle = load_bundle()
    m = bundle["model"]
    cases = [(7, 6, 2, 2), (7, 5.5, 2, 2), (6, 7, 2, 2), (8, 7, 2, 2), (8.5, 7, 2, 2), (5.0, 6.0, 0, 0), (10, 10, 0, 0)]
    for s, t, p, so in cases:
        app_pred = predict_category(s, t, p, so)
        direct = m.predict(pd.DataFrame([[s, t, p, so]], columns=FEATURES_A))[0]
        assert app_pred == direct, (s, t, p, so, app_pred, direct)
        print(f"  study={s} sleep={t} phys={p} social={so} -> {app_pred}")
    print("thresholds: OK")


def test_consistency_with_dataset_sample():
    df = load_data()
    bundle = load_bundle()
    sample = df.sample(20, random_state=7)
    mismatches = 0
    for _, r in sample.iterrows():
        app_pred = predict_category(r.Study_Hours_Per_Day, r.Sleep_Hours_Per_Day,
                                    r.Physical_Activity_Hours_Per_Day, r.Social_Hours_Per_Day)
        script_pred = bundle["model"].predict(r[FEATURES_A].to_frame().T.astype(float))[0]
        if app_pred != script_pred:
            mismatches += 1
    assert mismatches == 0
    print(f"consistency: {len(sample)} samples, {mismatches} mismatches")


def test_app_end_to_end():
    at = AppTest.from_file("app.py", default_timeout=30).run()
    assert at.session_state.page == "home"
    at.button[0].click().run()                      # Mulai pengisian
    assert at.session_state.page == "input"

    # Empty submit -> error, no result
    form_btn = [b for b in at.button if b.label == "Prediksi kategori"][0]
    form_btn.click().run()
    assert len(at.error) > 0, "expected error on empty input"
    assert at.session_state.result is None
    print("e2e empty input error: OK")

    # Over-24 total -> error
    for n, v in zip(at.number_input, [9.0, 9.0, 4.0, 3.0]):
        n.set_value(v)

    [b for b in at.button if b.label == "Prediksi kategori"][0].click().run()
    assert any("melebihi 24 jam" in e.value for e in at.error), [e.value for e in at.error]
    assert at.session_state.result is None
    print("e2e total>24 error: OK")

    # Valid input -> result page with category matching model
    at = AppTest.from_file("app.py", default_timeout=30).run()
    at.button[0].click().run()
    for n, v in zip(at.number_input, [7.0, 6.0, 2.0, 2.0]):
        n.set_value(v)

    [b for b in at.button if b.label == "Prediksi kategori"][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state.page == "result", at.session_state.page
    expected = predict_category(7.0, 6.0, 2.0, 2.0)
    assert at.session_state.result == expected
    assert any(expected in h.value for h in at.markdown)
    print("e2e valid prediction -> ", expected, ": OK")


if __name__ == "__main__":
    test_validation()
    test_thresholds_follow_model()
    test_consistency_with_dataset_sample()
    test_app_end_to_end()
    print("ALL TESTS PASSED")
