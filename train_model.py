"""Stage 2-3: split data, tune Decision Tree (CV on train only), train RF and
baseline, evaluate on test once, then retrain final Decision Tree on all data."""
import json
import joblib
import sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, confusion_matrix
from common import load_data, TARGET, CLASSES, FEATURES_A, FEATURES_B, RANDOM_STATE

MAX_DEPTHS = [2, 3, 4, 5, 6, 7, 8, 10, None]
N_FOLDS = 5


def split(df, features):
    X, y = df[features], df[TARGET]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)


def tune_tree(X_tr, y_tr):
    """Pick max_depth by 5-fold stratified CV on TRAIN data only (macro F1)."""
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        {"max_depth": MAX_DEPTHS},
        scoring="f1_macro",
        cv=cv,
        return_train_score=False,
    )
    gs.fit(X_tr, y_tr)
    results = pd.DataFrame(gs.cv_results_)[
        ["param_max_depth", "mean_test_score", "std_test_score", "mean_test_score"]
    ]
    cv_table = pd.DataFrame(gs.cv_results_)[["param_max_depth", "mean_test_score", "std_test_score"]]
    cv_table.columns = ["max_depth", "cv_macro_f1_mean", "cv_macro_f1_std"]
    cv_table["max_depth"] = cv_table["max_depth"].apply(lambda v: "None" if v is None or (isinstance(v, float) and np.isnan(v)) else int(v))
    return gs.best_params_["max_depth"], cv_table


def cv_accuracy(model, X_tr, y_tr):
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    res = cross_validate(model, X_tr, y_tr, cv=cv, scoring=["accuracy", "f1_macro"])
    return res["test_accuracy"].mean(), res["test_accuracy"].std(), res["test_f1_macro"].mean(), res["test_f1_macro"].std()


def metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)
    per_p, per_r, per_f, sup = precision_recall_fscore_support(y_true, y_pred, labels=CLASSES, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    return {
        "accuracy": acc, "macro_precision": p, "macro_recall": r, "macro_f1": f,
        "per_class": {c: {"precision": per_p[i], "recall": per_r[i], "f1": per_f[i], "support": int(sup[i])} for i, c in enumerate(CLASSES)},
        "confusion_matrix": cm,
    }


def run_scenario(name, features, df):
    X_tr, X_te, y_tr, y_te = split(df, features)
    best_depth, cv_table = tune_tree(X_tr, y_tr)
    dt_best = DecisionTreeClassifier(max_depth=best_depth, random_state=RANDOM_STATE)
    rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
    baseline = DummyClassifier(strategy="most_frequent")

    dt_acc_cv, dt_acc_sd, dt_f1_cv, dt_f1_sd = cv_accuracy(dt_best, X_tr, y_tr)
    rf_acc_cv, rf_acc_sd, rf_f1_cv, rf_f1_sd = cv_accuracy(rf, X_tr, y_tr)
    bl_acc_cv, bl_acc_sd, bl_f1_cv, bl_f1_sd = cv_accuracy(baseline, X_tr, y_tr)

    # Test evaluation (once, no tuning on test)
    dt_best.fit(X_tr, y_tr); rf.fit(X_tr, y_tr); baseline.fit(X_tr, y_tr)
    m_dt = metrics(y_te, dt_best.predict(X_te))
    m_rf = metrics(y_te, rf.predict(X_te))
    m_bl = metrics(y_te, baseline.predict(X_te))

    comparison = pd.DataFrame([
        {"scenario": name, "model": "Decision Tree (max_depth=%s)" % best_depth,
         "cv_accuracy": f"{dt_acc_cv:.3f} ± {dt_acc_sd:.3f}", "cv_macro_f1": f"{dt_f1_cv:.3f} ± {dt_f1_sd:.3f}",
         "test_accuracy": round(m_dt["accuracy"], 4), "test_macro_precision": round(m_dt["macro_precision"], 4),
         "test_macro_recall": round(m_dt["macro_recall"], 4), "test_macro_f1": round(m_dt["macro_f1"], 4)},
        {"scenario": name, "model": "Random Forest (n_estimators=300)",
         "cv_accuracy": f"{rf_acc_cv:.3f} ± {rf_acc_sd:.3f}", "cv_macro_f1": f"{rf_f1_cv:.3f} ± {rf_f1_sd:.3f}",
         "test_accuracy": round(m_rf["accuracy"], 4), "test_macro_precision": round(m_rf["macro_precision"], 4),
         "test_macro_recall": round(m_rf["macro_recall"], 4), "test_macro_f1": round(m_rf["macro_f1"], 4)},
        {"scenario": name, "model": "Baseline majority (High)",
         "cv_accuracy": f"{bl_acc_cv:.3f} ± {bl_acc_sd:.3f}", "cv_macro_f1": f"{bl_f1_cv:.3f} ± {bl_f1_sd:.3f}",
         "test_accuracy": round(m_bl["accuracy"], 4), "test_macro_precision": round(m_bl["macro_precision"], 4),
         "test_macro_recall": round(m_bl["macro_recall"], 4), "test_macro_f1": round(m_bl["macro_f1"], 4)},
    ])

    importance = pd.DataFrame({
        "feature": features,
        "decision_tree_importance": dt_best.feature_importances_,
        "random_forest_importance": rf.feature_importances_,
    }).sort_values("random_forest_importance", ascending=False)

    return {
        "name": name, "features": features, "best_depth": best_depth, "cv_table": cv_table,
        "comparison": comparison, "importance": importance,
        "dt_test": m_dt, "rf_test": m_rf, "bl_test": m_bl,
        "dt_model_split": dt_best,
    }


def main():
    df = load_data()
    print("Rows:", len(df), "| Features A:", FEATURES_A)

    res_A = run_scenario("A (lengkap)", FEATURES_A, df)
    res_B = run_scenario("B (tanpa belajar & tidur)", FEATURES_B, df)

    # Save results tables
    res_A["cv_table"].to_csv("reports/cv_max_depth_scenario_A.csv", index=False)
    res_B["cv_table"].to_csv("reports/cv_max_depth_scenario_B.csv", index=False)
    comp = pd.concat([res_A["comparison"], res_B["comparison"]], ignore_index=True)
    comp.to_csv("reports/model_comparison.csv", index=False)
    res_A["importance"].to_csv("reports/feature_importance_scenario_A.csv", index=False)
    res_B["importance"].to_csv("reports/feature_importance_scenario_B.csv", index=False)

    # Confusion matrices (test set)
    cm_rows = []
    for tag, res in [("A", res_A), ("B", res_B)]:
        for model_key, label in [("dt_test", "Decision Tree"), ("rf_test", "Random Forest")]:
            cm = res[model_key]["confusion_matrix"]
            for i, t in enumerate(CLASSES):
                for j, p in enumerate(CLASSES):
                    cm_rows.append({"scenario": tag, "model": label, "true": t, "pred": p, "count": int(cm[i, j])})
    pd.DataFrame(cm_rows).to_csv("reports/confusion_matrices_test.csv", index=False)

    # Final model: retrain Decision Tree on ALL 2000 rows, scenario A, with chosen depth
    final_depth = res_A["best_depth"]
    final = DecisionTreeClassifier(max_depth=final_depth, random_state=RANDOM_STATE)
    final.fit(df[FEATURES_A], df[TARGET])
    bundle = {
        "model": final,
        "feature_names": FEATURES_A,
        "classes": list(final.classes_),
        "config": {"algorithm": "DecisionTreeClassifier", "max_depth": final_depth,
                   "random_state": RANDOM_STATE, "trained_on": "all 2000 rows, scenario A"},
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(bundle, "models/decision_tree_model.joblib")

    summary = {
        "best_depth_A": res_A["best_depth"], "best_depth_B": res_B["best_depth"],
        "final_model_classes": list(final.classes_), "sklearn_version": sklearn.__version__,
    }
    with open("reports/run_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Print for the report
    pd.set_option("display.width", 200)
    for res in [res_A, res_B]:
        print("\n===", res["name"], "===")
        print("CV max_depth table:\n", res["cv_table"].to_string(index=False))
        print("Best max_depth:", res["best_depth"])
        print(res["comparison"].to_string(index=False))
        print("Feature importance:\n", res["importance"].to_string(index=False))
        for key, lab in [("dt_test", "DT"), ("rf_test", "RF")]:
            m = res[key]
            print(f"{lab} per-class test:", {c: {k: round(v, 3) if isinstance(v, float) else v for k, v in d.items()} for c, d in m["per_class"].items()})
            print(f"{lab} confusion (rows=true {CLASSES}):\n", m["confusion_matrix"])
    print("\nFinal DT depth:", final_depth, "| classes:", list(final.classes_), "| sklearn:", sklearn.__version__)


if __name__ == "__main__":
    main()
