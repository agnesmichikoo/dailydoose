"""Stage 4: evaluation outputs (per-scenario CSV, confusion matrix PNG,
feature importance PNG). Reuses the same pipeline as train_model.py."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from common import load_data, CLASSES, FEATURES_A, FEATURES_B
from train_model import run_scenario


def per_class_rows(scenario_name, res):
    rows = []
    for key, label in [("dt_test", "Decision Tree"), ("rf_test", "Random Forest"), ("bl_test", "Baseline (High)")]:
        m = res[key]
        for c in CLASSES:
            d = m["per_class"][c]
            rows.append({"scenario": scenario_name, "model": label, "class": c,
                         "precision": round(float(d["precision"]), 4), "recall": round(float(d["recall"]), 4),
                         "f1": round(float(d["f1"]), 4), "support": d["support"], "accuracy": ""})
        rows.append({"scenario": scenario_name, "model": label, "class": "MACRO (rata-rata 3 kelas)",
                     "precision": round(float(m["macro_precision"]), 4), "recall": round(float(m["macro_recall"]), 4),
                     "f1": round(float(m["macro_f1"]), 4), "support": int(m["confusion_matrix"].sum()),
                     "accuracy": round(float(m["accuracy"]), 4)})
    return rows


def plot_cm(cm, title, path):
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(CLASSES); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual"); ax.set_title(title)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_importance(imp, title, path):
    imp = imp.sort_values("decision_tree_importance")
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.barh(imp["feature"], imp["decision_tree_importance"], color="#4F6BED")
    ax.set_xlabel("Kontribusi relatif (Decision Tree, model final)")
    ax.set_title(title); ax.set_xlim(0, 1)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    df = load_data()
    res_A = run_scenario("A (lengkap)", FEATURES_A, df)
    res_B = run_scenario("B (tanpa belajar & tidur)", FEATURES_B, df)

    pd.DataFrame(per_class_rows("A", res_A)).to_csv("reports/evaluation_scenario_A.csv", index=False)
    pd.DataFrame(per_class_rows("B", res_B)).to_csv("reports/evaluation_scenario_B.csv", index=False)

    plot_cm(res_A["dt_test"]["confusion_matrix"], "Confusion matrix: Decision Tree\nSkenario A (test)", "reports/confusion_matrix_A_decision_tree.png")
    plot_cm(res_B["dt_test"]["confusion_matrix"], "Confusion matrix: Decision Tree\nSkenario B (test)", "reports/confusion_matrix_B_decision_tree.png")
    plot_cm(res_A["rf_test"]["confusion_matrix"], "Confusion matrix: Random Forest\nSkenario A (test)", "reports/confusion_matrix_A_random_forest.png")
    plot_cm(res_B["rf_test"]["confusion_matrix"], "Confusion matrix: Random Forest\nSkenario B (test)", "reports/confusion_matrix_B_random_forest.png")

    # Global feature importance of the deployed model (Decision Tree, scenario A)
    plot_importance(res_A["importance"], "Feature importance global: model Decision Tree (skenario A)", "reports/feature_importance.png")

    # Copy the main confusion matrix under the spec's expected name
    import shutil
    shutil.copy("reports/confusion_matrix_A_decision_tree.png", "reports/confusion_matrix.png")
    print("Evaluation files written.")


if __name__ == "__main__":
    main()
