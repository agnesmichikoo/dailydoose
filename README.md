# Daily Doose: Prediksi Kategori Stres Mahasiswa

Prototipe aplikasi web untuk memprediksi kategori stres mahasiswa (**Low**, **Moderate**, **High**) berdasarkan empat kebiasaan harian. Dibuat sebagai luaran PKM-KC (Program Kreativitas Mahasiswa bidang Karsa Cipta), dengan pendekatan integrasi Artificial Intelligence dan Psikologi Kognitif.

> **Batasan penting:** aplikasi ini bukan alat diagnosis psikologis, bukan pengukuran klinis stres, dan tidak menggantikan konsultasi dengan psikolog atau tenaga profesional. Kategori yang ditampilkan mengikuti definisi kategori dalam Student Lifestyle Dataset, bukan pengukuran stres psikologis seseorang.

## Fitur model

| Input | Rentang | Satuan |
|---|---|---|
| Durasi belajar | 5–10 | jam per hari |
| Durasi tidur | 5–10 | jam per hari |
| Aktivitas fisik | 0–13 | jam per hari |
| Aktivitas sosial | 0–6 | jam per hari |

Model utama: **Decision Tree** (`max_depth=3`), dilatih ulang pada seluruh 2.000 baris data. Model pembanding: Random Forest (`n_estimators=300`) dan baseline mayoritas (selalu High).

## Struktur proyek

```
student-stress-app/
├── app.py              # Antarmuka Streamlit
├── predictor.py        # Memuat model Joblib dan memanggil model.predict()
├── validation.py       # Validasi input (rentang, negatif, total > 24 jam)
├── common.py           # Konstanta fitur dan pemuatan data
├── train_model.py      # Preprocessing, pembagian data, tuning, pelatihan, evaluasi
├── evaluate_model.py   # Tabel evaluasi dan grafik untuk laporan
├── test_app.py         # Pengujian validasi, ambang, konsistensi, dan end-to-end
├── student_lifestyle_dataset.csv
├── requirements.txt
├── models/decision_tree_model.joblib
├── reports/            # Hasil eksekusi skrip (CSV dan PNG)
└── assets/logo.png
```

## Cara menjalankan

Proyek diuji dengan **Python 3.12**.

```bash
# 1. Buat dan aktifkan virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Pasang dependensi
pip install -r requirements.txt

# 3. (Opsional) Latih ulang model dan buat laporan evaluasi
python train_model.py
python evaluate_model.py

# 4. (Opsional) Jalankan pengujian
python test_app.py

# 5. Jalankan aplikasi
streamlit run app.py
```

Aplikasi akan terbuka di `http://localhost:8501`.

## Versi perangkat lunak

- Scikit-learn **1.9.1** (versi yang digunakan saat model disimpan; tercatat juga di dalam file `.joblib`)
- Streamlit **1.67.0**
- Pandas, NumPy, Joblib, dan Matplotlib: lihat `requirements.txt`

Model `.joblib` sebaiknya dimuat dengan versi scikit-learn yang sama. Jika versi berbeda, latih ulang dengan `python train_model.py`.

## Metodologi singkat

1. **Preprocessing:** menghapus `Student_ID`, `GPA`, dan `Extracurricular_Hours_Per_Day`. Tidak ada missing value dan tidak ada baris duplikat.
2. **Pembagian data:** 80% train dan 20% test, stratified, `random_state=42`.
3. **Pemilihan model:** `max_depth` Decision Tree dipilih dengan 5-fold stratified cross-validation pada data train, memakai Macro F1. Data test tidak digunakan untuk tuning. Jika beberapa kedalaman memberi skor yang sama, dipilih yang paling sederhana.
4. **Evaluasi akhir:** dilakukan sekali pada data test. Model final dilatih ulang pada seluruh 2.000 baris.

Verifikasi aturan label: aturan High, Low, dan Moderate yang tercatat di draf cocok dengan seluruh 2.000 baris. Ini adalah hasil pemeriksaan data, bukan dokumentasi resmi dataset.

## Hasil ringkas (data test, 400 baris)

| Model | Skenario | Accuracy | Macro F1 |
|---|---|---|---|
| Decision Tree | A (lengkap) | 1,000 | 1,000 |
| Random Forest | A (lengkap) | 1,000 | 1,000 |
| Baseline (selalu High) | A dan B | 0,515 | 0,227 |
| Decision Tree | B (tanpa belajar & tidur) | 0,405 | 0,349 |
| Random Forest | B (tanpa belajar & tidur) | 0,410 | 0,345 |

Skor sempurna pada skenario A **tidak** menunjukkan kemampuan memprediksi stres psikologis. Label dibentuk langsung dari durasi belajar dan tidur, sehingga model hanya mempelajari aturan tersebut. Pada skenario B, accuracy di bawah baseline, meskipun Macro F1 sedikit di atas baseline.

Rincian lengkap ada di `reports/`.

## Sumber dataset

- Student Lifestyle Dataset, Kaggle, DOI: `10.34740/kaggle/ds/6063456`, diunggah oleh akun `steve1215rogg`.
- Nama asli pemilik dataset **belum diverifikasi** dan tidak dicantumkan.
- Tahun unggah, URL lengkap, dan tanggal akses **masih perlu dilengkapi**.
- Asal-usul kolom `Stress_Level` **belum ditemukan dokumentasinya**.

## Keterbatasan

- Label kategori dibentuk dari durasi belajar dan tidur, sehingga validitas prediksi terbatas pada definisi dataset.
- Data berupa laporan diri (self-report) dan tidak memuat instrumen stres yang tervalidasi.
- Sampel didominasi mahasiswa dari India (2023–2024), sehingga generalisasi ke mahasiswa Indonesia perlu hati-hati.
- Setiap baris menjumlahkan tepat 24 jam untuk lima kolom aktivitas, yang menunjukkan kemungkinan alokasi waktu tetap, bukan catatan harian yang independen. Klaim "320 baris melebihi 24 jam" dari draf tidak didukung oleh berkas CSV ini.
- Aplikasi hanya meminta durasi tidur, bukan kualitas tidur.
- Feature importance menunjukkan kontribusi relatif fitur pada model secara keseluruhan, bukan bukti hubungan sebab-akibat.

## Layanan konseling kampus

Tautan layanan konseling kampus belum tersedia di aplikasi. Isi variabel `COUNSELING_URL` di `app.py` setelah alamat resminya diperoleh.

## Deployment

Lihat `DEPLOYMENT.md`.

## Lisensi dan penggunaan

Prototipe akademik untuk keperluan edukasi dan demonstrasi PKM-KC.
