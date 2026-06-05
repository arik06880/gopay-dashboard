"""
Empat bagian halaman dashboard. Setiap fungsi merender satu tab dan menangani
sendiri kasus berkas yang belum ada agar halaman tidak menabrak galat.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from . import theme
from .loaders import load_csv, load_dataset, load_lda, load_model
from .preprocessing import preprocess

KELAS_URUT = ["positif", "negatif", "netral"]


# ---------------------------------------------------------------------------
# Util
# ---------------------------------------------------------------------------
def _baca_metrik(eval_df):
    """Ambil accuracy dan F1-macro dari hasil_evaluasi_nb.csv secara toleran."""
    if eval_df is None or eval_df.empty:
        return None, None
    df = eval_df.copy()
    df = df.set_index(df.columns[0])
    df.index = df.index.astype(str).str.strip()

    def _nilai(baris, kolom_utama="f1-score"):
        if baris not in df.index:
            return None
        row = df.loc[baris]
        if kolom_utama in df.columns and pd.notna(row.get(kolom_utama)):
            return float(row[kolom_utama])
        for v in row:
            try:
                if pd.notna(v):
                    return float(v)
            except (TypeError, ValueError):
                continue
        return None

    return _nilai("accuracy"), _nilai("macro avg")


# ---------------------------------------------------------------------------
# 1. Ringkasan
# ---------------------------------------------------------------------------
def ringkasan():
    df = load_dataset()
    eval_df = load_csv("hasil_evaluasi_nb.csv")
    diskrim = load_csv("kata_diskriminatif.csv")

    st.subheader("Ringkasan")

    total = f"{len(df):,}".replace(",", ".") if df is not None else "-"
    acc, f1m = _baca_metrik(eval_df)
    acc_s = f"{acc*100:.1f}%" if acc is not None else "-"
    f1_s = f"{f1m:.3f}" if f1m is not None else "-"

    theme.cards([
        ("Total ulasan", total, "setelah preprocessing"),
        ("Akurasi model", acc_s, "Naive Bayes pada data uji"),
        ("F1-macro", f1_s, "rata-rata tiga kelas"),
    ])

    if df is not None and "label" in df.columns:
        dist = (
            df["label"].value_counts()
            .reindex(KELAS_URUT).dropna()
            .rename_axis("kelas").reset_index(name="jumlah")
        )
        fig = px.pie(
            dist, names="kelas", values="jumlah", hole=0.55,
            color="kelas", color_discrete_map=theme.SENT,
        )
        fig.update_traces(textinfo="percent+label", sort=False)
        st.plotly_chart(theme.plotly_layout(fig), use_container_width=True)
    else:
        theme.note("Unggah <code>data/gopay_preprocessed.csv</code> untuk menampilkan distribusi sentimen.")

    if diskrim is not None and {"kelas", "kata"}.issubset(diskrim.columns):
        st.markdown("**Kata paling khas tiap kelas**")
        for k in KELAS_URUT:
            sub = diskrim[diskrim["kelas"].str.lower() == k]
            if not sub.empty:
                theme.badge(k)
                theme.chips(sub["kata"].tolist())
                st.write("")


# ---------------------------------------------------------------------------
# 2. Prediksi langsung
# ---------------------------------------------------------------------------
def prediksi():
    tfidf, model = load_model()

    st.subheader("Prediksi langsung")
    st.write("Ketik sebuah ulasan, lalu model memperkirakan sentimennya beserta probabilitas tiap kelas.")

    if tfidf is None or model is None:
        theme.note("Letakkan <code>tfidf_vectorizer.pkl</code> dan <code>model_nb.pkl</code> di folder <code>models/</code> untuk mengaktifkan fitur ini.")
        return

    teks = st.text_area("Teks ulasan", height=120, placeholder="Contoh: aplikasinya bagus, transfer cepat dan mudah")
    if st.button("Prediksi"):
        bersih = preprocess(teks)
        if not bersih:
            st.warning("Teks kosong setelah preprocessing. Coba tulis ulasan yang lebih panjang.")
            return
        X = tfidf.transform([bersih])
        proba = model.predict_proba(X)[0]
        kelas = list(model.classes_)
        pred = kelas[int(np.argmax(proba))]

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("**Prediksi**")
            theme.badge(pred)
        with col2:
            prob_df = pd.DataFrame({
                "kelas": kelas,
                "probabilitas": proba,
            }).sort_values("probabilitas", ascending=True)
            fig = px.bar(
                prob_df, x="probabilitas", y="kelas", orientation="h",
                color="kelas", color_discrete_map=theme.SENT, text_auto=".1%",
            )
            fig.update_layout(showlegend=False)
            fig.update_xaxes(range=[0, 1], tickformat=".0%")
            st.plotly_chart(theme.plotly_layout(fig), use_container_width=True)

        with st.expander("Lihat teks setelah preprocessing"):
            st.code(bersih or "(kosong)")

        if pred == "netral" or proba[kelas.index("netral")] > 0.33:
            theme.note("Prediksi kelas netral kurang andal: model memiliki <em>recall</em> rendah pada kelas ini karena ulasan netral beririsan secara leksikal dengan kelas lain. Perlakukan hasil netral dengan hati-hati.")


# ---------------------------------------------------------------------------
# 3. Topik LDA
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _proporsi_topik(kelas: str):
    """Hitung proporsi topik dominan untuk satu kelas dari model LDA tersimpan."""
    bundle = load_lda(kelas)
    df = load_dataset()
    if bundle is None or df is None or "teks_bersih" not in df.columns:
        return None
    docs = df.loc[df["label"] == kelas, "teks_bersih"].dropna().astype(str)
    docs = docs[docs.str.strip() != ""]
    if docs.empty:
        return None
    cv, lda = bundle["cv"], bundle["lda"]
    X = cv.transform(docs)
    theta = lda.transform(X)
    dom = theta.argmax(axis=1)
    n = lda.n_components
    cnt = np.bincount(dom, minlength=n).astype(float)
    prop = cnt / cnt.sum() * 100
    return pd.DataFrame({"topik_id": np.arange(1, n + 1), "proporsi": prop})


def topik():
    topik_df = load_csv("hasil_topik_lda.csv")
    coh = load_csv("coherence_scores.csv")

    st.subheader("Topik LDA per kelas")
    if topik_df is None:
        theme.note("Unggah <code>data/hasil_topik_lda.csv</code> untuk menampilkan topik.")
        return

    kelas = st.selectbox("Kelas sentimen", KELAS_URUT, format_func=lambda k: theme.SENT_LABEL[k])

    if coh is not None and kelas in coh.columns:
        baris_terbaik = coh[kelas].astype(float).idxmax()
        k_terpilih = int(coh.loc[baris_terbaik, "k"])
        skor = float(coh.loc[baris_terbaik, kelas])
        theme.note(f"Jumlah topik terpilih untuk kelas {theme.SENT_LABEL[kelas].lower()}: <strong>k = {k_terpilih}</strong> "
                   f"berdasarkan <em>coherence</em> c_v tertinggi ({skor:.4f}).")

    sub = topik_df[topik_df["kelas"].str.lower() == kelas].copy()
    prop = _proporsi_topik(kelas)
    if prop is not None:
        sub = sub.merge(prop, on="topik_id", how="left")
        fig = px.bar(
            sub.sort_values("proporsi"), x="proporsi", y="nama_topik",
            orientation="h", text_auto=".1f",
            color_discrete_sequence=[theme.SENT[kelas]],
        )
        fig.update_layout(showlegend=False)
        fig.update_xaxes(title="Proporsi dokumen (%)")
        fig.update_yaxes(title="")
        st.plotly_chart(theme.plotly_layout(fig), use_container_width=True)
    else:
        theme.note("Letakkan model <code>lda_*.pkl</code> di folder <code>models/</code> untuk menampilkan proporsi topik. Tabel kata kunci tetap tersedia di bawah.")

    tampil = sub[["topik_id", "nama_topik", "kata_kunci"]].rename(
        columns={"topik_id": "Topik", "nama_topik": "Nama topik", "kata_kunci": "Kata kunci"}
    )
    st.dataframe(tampil, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# 4. Eksplorasi ulasan
# ---------------------------------------------------------------------------
def eksplorasi():
    df = load_dataset()
    st.subheader("Eksplorasi ulasan")
    if df is None:
        theme.note("Unggah <code>data/gopay_preprocessed.csv</code> untuk menjelajahi ulasan.")
        return

    c1, c2 = st.columns([2, 1])
    with c1:
        pilih_kelas = st.multiselect(
            "Kelas", KELAS_URUT, default=KELAS_URUT,
            format_func=lambda k: theme.SENT_LABEL[k],
        )
    with c2:
        bulan_opsi = ["Semua"]
        if "bulan" in df.columns:
            bulan_opsi += sorted(df["bulan"].dropna().unique().tolist())
        pilih_bulan = st.selectbox("Bulan", bulan_opsi)

    mask = df["label"].isin(pilih_kelas)
    if pilih_bulan != "Semua" and "bulan" in df.columns:
        mask &= df["bulan"] == pilih_bulan
    hasil = df[mask]

    # Tren temporal
    if "bulan" in df.columns:
        tren = (
            df[df["label"].isin(pilih_kelas)]
            .groupby(["bulan", "label"]).size().reset_index(name="jumlah")
        )
        fig = px.line(
            tren, x="bulan", y="jumlah", color="label", markers=True,
            color_discrete_map=theme.SENT,
        )
        fig.update_xaxes(title="")
        fig.update_yaxes(title="Jumlah ulasan")
        st.plotly_chart(theme.plotly_layout(fig), use_container_width=True)

    st.markdown(f"**{len(hasil):,} ulasan** sesuai penyaring.".replace(",", "."))
    kolom = [c for c in ["ulasan", "label", "bintang", "tanggal"] if c in hasil.columns]
    st.dataframe(hasil[kolom].head(300), use_container_width=True, hide_index=True)
    if len(hasil) > 300:
        theme.note("Menampilkan 300 baris pertama untuk menjaga kecepatan.")
