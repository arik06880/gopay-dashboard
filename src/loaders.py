"""
Pemuat artefak: model, vectorizer, model LDA, berkas CSV, dan dataset.

Semua dibungkus cache Streamlit agar hanya dimuat sekali. Berkas model dimuat
sebagai sumber daya (cache_resource), berkas tabel sebagai data (cache_data).
Jika berkas belum ada, fungsi mengembalikan None sehingga halaman dapat
menampilkan pesan ramah, bukan menabrak galat.
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
DATA = ROOT / "data"

KELAS = ("positif", "negatif", "netral")


# ---------------------------------------------------------------------------
# Model & vectorizer
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    """Kembalikan (tfidf, model) atau (None, None) bila berkas belum ada."""
    f_tfidf = MODELS / "tfidf_vectorizer.pkl"
    f_model = MODELS / "model_nb.pkl"
    if not (f_tfidf.exists() and f_model.exists()):
        return None, None
    return joblib.load(f_tfidf), joblib.load(f_model)


@st.cache_resource(show_spinner=False)
def load_lda(kelas: str):
    """Kembalikan bundel {'cv', 'lda'} untuk satu kelas, atau None."""
    f = MODELS / f"lda_{kelas}.pkl"
    if not f.exists():
        return None
    return joblib.load(f)


# ---------------------------------------------------------------------------
# Berkas CSV hasil
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(nama: str):
    """Baca berkas CSV dari folder data, atau None bila tidak ada."""
    f = DATA / nama
    if not f.exists():
        return None
    return pd.read_csv(f)


@st.cache_data(show_spinner=False)
def load_dataset():
    """
    Baca gopay_preprocessed.csv. Kolom nama dibuang secara defensif bila ada,
    dan kolom tanggal diubah ke tipe datetime serta diberi kolom bulan.
    """
    f = DATA / "gopay_preprocessed.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    if "nama" in df.columns:
        df = df.drop(columns=["nama"])
    if "tanggal" in df.columns:
        df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")
        df["bulan"] = df["tanggal"].dt.to_period("M").astype(str)
    return df


def cek_kelengkapan():
    """Ringkas berkas wajib mana yang sudah ada, untuk panel status."""
    wajib = {
        "models/tfidf_vectorizer.pkl": (MODELS / "tfidf_vectorizer.pkl").exists(),
        "models/model_nb.pkl": (MODELS / "model_nb.pkl").exists(),
        "models/lda_positif.pkl": (MODELS / "lda_positif.pkl").exists(),
        "models/lda_negatif.pkl": (MODELS / "lda_negatif.pkl").exists(),
        "models/lda_netral.pkl": (MODELS / "lda_netral.pkl").exists(),
        "data/gopay_preprocessed.csv": (DATA / "gopay_preprocessed.csv").exists(),
        "data/hasil_evaluasi_nb.csv": (DATA / "hasil_evaluasi_nb.csv").exists(),
        "data/hasil_topik_lda.csv": (DATA / "hasil_topik_lda.csv").exists(),
        "data/coherence_scores.csv": (DATA / "coherence_scores.csv").exists(),
        "data/kata_diskriminatif.csv": (DATA / "kata_diskriminatif.csv").exists(),
    }
    return wajib
