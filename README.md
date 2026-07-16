# Dashboard Sentimen GoPay

Purwarupa dashboard untuk skripsi analisis sentimen ulasan GoPay di Google Play
Store (Januari - Maret 2026) dengan Naive Bayes dan LDA. Dashboard hanya memuat
model serta berkas hasil yang sudah disimpan, tidak melatih ulang, sehingga
angka yang ditampilkan konsisten dengan laporan.

## Struktur proyek

```
gopay-dashboard/
├── app.py                 # titik masuk aplikasi
├── requirements.txt       # dependensi
├── .streamlit/config.toml # tema
├── src/
│   ├── preprocessing.py   # pipeline preprocessing identik notebook 02
│   ├── loaders.py         # pemuat model & berkas
│   ├── theme.py           # visual & komponen
│   └── sections.py        # empat bagian halaman
├── models/                # model
└── data/                  # data
```


