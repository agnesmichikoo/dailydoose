# Panduan Deployment ke Streamlit Community Cloud

Panduan ini membuat aplikasi dapat diakses lewat tautan publik. Langkah-langkah di bawah dijalankan oleh kamu sendiri, karena membutuhkan akun GitHub dan Streamlit.

## Prasyarat

- Akun GitHub
- Akun Streamlit Community Cloud (login dengan GitHub di https://share.streamlit.io)
- Folder `student-stress-app/` lengkap, termasuk `models/decision_tree_model.joblib`

## Langkah 1: Cek file sebelum upload

Pastikan file berikut ada:
- `app.py`, `predictor.py`, `validation.py`, `common.py`
- `requirements.txt`
- `models/decision_tree_model.joblib`
- `README.md`

Jalankan pengujian di komputer lokal sebelum upload:

```bash
python test_app.py
```

Semua tes harus lulus (`ALL TESTS PASSED`).

## Langkah 2: Buat repositori GitHub

1. Buka https://github.com/new
2. Beri nama repositori, misalnya `student-stress-app`.
3. Pilih **Public**, agar Streamlit Community Cloud dapat mengaksesnya.
4. Jangan centang "Add a README" karena README sudah ada.
5. Klik **Create repository**.

## Langkah 3: Upload kode

Cara paling mudah lewat terminal:

```bash
cd student-stress-app
git init
git add .
git commit -m "Initial commit: prototipe prediksi kategori stres"
git branch -M main
git remote add origin https://github.com/USERNAME/student-stress-app.git
git push -u origin main
```

Ganti `USERNAME` dengan nama akun GitHub kamu.

Jika tidak memakai terminal, kamu juga bisa drag and drop semua file ke halaman repositori di GitHub.

> Catatan: file `.joblib` berukuran kecil sehingga aman di-upload ke GitHub biasa. Jika kamu menambahkan file `.venv/` atau `__pycache__/`, abaikan dengan membuat berkas `.gitignore` berisi baris berikut:
>
> ```
> .venv/
> __pycache__/
> ```

## Langkah 4: Deploy di Streamlit Community Cloud

1. Buka https://share.streamlit.io dan login dengan GitHub.
2. Klik **Create app** lalu pilih **Deploy a public app from GitHub**.
3. Isi kolom berikut:
   - **Repository:** `USERNAME/student-stress-app`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **Python version:** pilih **3.12**
4. Klik **Deploy**.

Proses instalasi biasanya memakan waktu beberapa menit. Streamlit akan membaca `requirements.txt` secara otomatis.

## Langkah 5: Verifikasi setelah deploy

Jangan menganggap aplikasi sudah online sebelum kamu mengujinya sendiri. Periksa:

1. Halaman beranda tampil dengan benar.
2. Formulir menerima input dan menolak input kosong atau di luar rentang.
3. Input dengan total lebih dari 24 jam ditolak dengan pesan yang sesuai.
4. Input valid menampilkan kategori dan peringatan.
5. Tautan dapat dibuka dari perangkat lain (misalnya ponsel dengan data seluler).

Jika salah satu gagal, cek log di pojok kanan bawah aplikasi (**Manage app → Logs**).

## Langkah 6: Salin tautan publik

Tautan aplikasi berbentuk `https://NAMA-APP.streamlit.app`. Simpan tautan ini untuk laporan dan media sosial PKM-KC.

## Masalah umum

- **Error saat memuat model `.joblib`:** versi scikit-learn di server berbeda. Pastikan `requirements.txt` memakai `scikit-learn==1.9.1`. Jika versi tersebut belum tersedia di server, jalankan `python train_model.py` lokal lalu upload ulang model.
- **Aplikasi "sleeping":** Streamlit Community Cloud menghentikan aplikasi yang lama tidak dikunjungi. Buka tautannya lagi untuk menyalakannya kembali.
- **Perubahan kode tidak muncul:** push ke GitHub, lalu klik **Reboot app** di dashboard Streamlit.
- **Tautan konseling kosong:** isi `COUNSELING_URL` di `app.py`, lalu push ulang.

## Pembaruan setelah deploy

Setiap kali kode di branch `main` diubah dan di-push, Streamlit akan memperbarui aplikasi secara otomatis.
