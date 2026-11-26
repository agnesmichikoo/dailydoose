"""Input validation (Tahap 6). Returns a list of error messages; empty list = valid."""
from typing import Optional

# (key, label, min, max)
INPUT_SPECS = [
    ("study", "durasi belajar", 5.0, 10.0),
    ("sleep", "durasi tidur", 5.0, 10.0),
    ("physical", "aktivitas fisik", 0.0, 13.0),
    ("social", "aktivitas sosial", 0.0, 6.0),
]
MAX_TOTAL_HOURS = 24.0


def validate_inputs(values: dict) -> list:
    errors = []
    missing = [label for key, label, _, _ in INPUT_SPECS if values.get(key) is None]
    if missing:
        return ["Semua kolom wajib diisi. Belum diisi: " + ", ".join(missing) + "."]

    for key, label, lo, hi in INPUT_SPECS:
        v = float(values[key])
        if v < 0:
            errors.append(f"Nilai {label} tidak boleh negatif.")
        elif v < lo or v > hi:
            errors.append(f"Nilai {label} harus antara {lo:g} dan {hi:g} jam per hari.")

    if not errors:
        total = sum(float(values[k]) for k, _, _, _ in INPUT_SPECS)
        if total > MAX_TOTAL_HOURS:
            errors.append("Total durasi aktivitas yang kamu masukkan melebihi 24 jam. Silakan periksa kembali jawabanmu.")
    return errors
