# Dashboard Sentimen GoPay

Purwarupa dashboard untuk skripsi analisis sentimen ulasan GoPay di Google Play
Store (Januari–Maret 2026) dengan Naive Bayes dan LDA. Dashboard hanya memuat
model serta berkas hasil yang sudah disimpan, tidak melatih ulang, sehingga
angka yang ditampilkan konsisten dengan laporan.

## Struktur proyek

```
gopay-dashboard/
├── app.py                 # titik masuk aplikasi
├── requirements.txt       # dependensi (PERHATIKAN penguncian versi)
├── .streamlit/config.toml # tema
├── src/
│   ├── preprocessing.py   # pipeline preprocessing identik notebook 02
│   ├── loaders.py         # pemuat model & berkas (dengan cache)
│   ├── theme.py           # gaya visual & komponen
│   └── sections.py        # empat bagian halaman
├── models/                # letakkan berkas .pkl di sini
└── data/                  # letakkan berkas .csv di sini
```

## Berkas yang harus Anda siapkan

Letakkan di folder `models/`:

| Berkas | Asal | Kegunaan |
| --- | --- | --- |
| `model_nb.pkl` | notebook 03 | model Naive Bayes untuk prediksi langsung |
| `tfidf_vectorizer.pkl` | notebook 03 | mengubah teks menjadi fitur TF-IDF |
| `lda_positif.pkl` | notebook 04 | proporsi topik kelas positif |
| `lda_negatif.pkl` | notebook 04 | proporsi topik kelas negatif |
| `lda_netral.pkl` | notebook 04 | proporsi topik kelas netral |

Letakkan di folder `data/`:

| Berkas | Asal | Kegunaan |
| --- | --- | --- |
| `gopay_preprocessed.csv` | notebook 02 | distribusi, eksplorasi, tren, sumber dokumen topik |
| `hasil_evaluasi_nb.csv` | notebook 03 | akurasi dan F1-macro pada ringkasan |
| `hasil_topik_lda.csv` | notebook 04 | nama topik dan kata kunci |
| `coherence_scores.csv` | notebook 04 | nilai k terpilih per kelas |
| `kata_diskriminatif.csv` | notebook 05 | kata khas per kelas |

Berkas opsional (boleh ditambahkan, tidak wajib): `hasil_gridsearch.csv`,
`hasil_cv.csv`, `hasil_prediksi_test.csv`, `sampel_misklasifikasi.csv`.

### Privasi
Sebelum mengunggah `gopay_preprocessed.csv` ke repositori publik, hapus kolom
nama pengguna. Aplikasi juga membuang kolom `nama` secara otomatis bila masih
ada, tetapi sebaiknya tidak ikut diunggah sejak awal.

## Catatan versi (penting)

Berkas `.pkl` hanya dapat dimuat bila versi scikit-learn, numpy, dan scipy di
`requirements.txt` sama dengan versi saat melatih di Colab. Jalankan di Colab:

```python
import sklearn, numpy, scipy
print(sklearn.__version__, numpy.__version__, scipy.__version__)
```

lalu samakan ketiga baris pada `requirements.txt`. Versi Sastrawi juga harus
sama agar hasil stemming identik.

## Menjalankan secara lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy ke streamlit.io

1. Unggah seluruh folder ini ke repositori GitHub publik (sertakan berkas di
   `models/` dan `data/`).
2. Buka share.streamlit.io, hubungkan akun GitHub, pilih repositori, dan
   tetapkan `app.py` sebagai berkas utama.
3. Streamlit akan memasang dependensi dan menjalankan aplikasi.
