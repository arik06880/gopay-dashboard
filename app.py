"""
Dashboard analisis sentimen ulasan GoPay (Naive Bayes + LDA).

Purwarupa untuk menyajikan hasil penelitian secara interaktif: ringkasan,
prediksi langsung, topik LDA, dan eksplorasi ulasan. Aplikasi hanya memuat
model serta berkas hasil yang sudah disimpan, tidak melatih ulang, sehingga
angka yang ditampilkan konsisten dengan laporan skripsi.
"""

import streamlit as st

from src import sections, theme
from src.loaders import cek_kelengkapan

st.set_page_config(
    page_title="Sentimen GoPay",
    page_icon="•",
    layout="wide",
    initial_sidebar_state="collapsed",
)

theme.inject()

theme.cover(
    eyebrow="Analisis Sentimen · Google Play Store · Jan–Mar 2026",
    judul="Suara Pengguna GoPay",
    lede="Purwarupa dashboard untuk menelusuri sentimen dan topik ulasan GoPay. "
         "Model Naive Bayes mengklasifikasikan sentimen, sedangkan LDA memetakan "
         "tema yang dibicarakan pengguna pada tiap kelas.",
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Ringkasan", "Prediksi langsung", "Topik LDA", "Eksplorasi ulasan"]
)

with tab1:
    sections.ringkasan()
with tab2:
    sections.prediksi()
with tab3:
    sections.topik()
with tab4:
    sections.eksplorasi()

# Panel status berkas di sidebar
with st.sidebar:
    st.markdown("### Status berkas")
    for nama, ada in cek_kelengkapan().items():
        st.write(("✓ " if ada else "✗ ") + nama)
    st.caption(
        "Tanda ✗ berarti berkas belum diletakkan. Bagian terkait akan "
        "menampilkan pesan, bagian lain tetap berjalan."
    )
