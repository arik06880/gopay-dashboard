"""
Preprocessing teks untuk dashboard.

Modul ini menyalin persis pipeline dari notebook 02 agar teks yang diketik
pengguna diproses dengan cara yang sama seperti data latih. Jika satu langkah
berbeda, prediksi akan menyimpang dari distribusi yang dipelajari model.
"""

import re
import nltk
from functools import lru_cache
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Unduh resource NLTK secara otomatis dan senyap untuk kebutuhan Streamlit Cloud
try:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
except Exception:
    pass

from nltk.tokenize import word_tokenize

# ---------------------------------------------------------------------------
# Kamus slang — disalin persis dari notebook 02
# ---------------------------------------------------------------------------
KAMUS_SLANG = {
    'gk': 'tidak', 'ga': 'tidak', 'gak': 'tidak', 'ngga': 'tidak',
    'nggak': 'tidak', 'tdk': 'tidak', 'kaga': 'tidak', 'kagak': 'tidak',
    'ko': 'kok', 'bgt': 'sangat', 'banget': 'sangat', 'bngt': 'sangat',
    'sy': 'saya', 'aq': 'saya', 'gw': 'saya', 'gue': 'saya', 'gua': 'saya',
    'sya': 'saya', 'km': 'kamu', 'lo': 'kamu', 'lu': 'kamu',
    'yg': 'yang', 'dg': 'dengan', 'dgn': 'dengan', 'dr': 'dari',
    'utk': 'untuk', 'tuk': 'untuk', 'krn': 'karena', 'karna': 'karena',
    'tp': 'tapi', 'tpi': 'tapi', 'pdhl': 'padahal', 'sm': 'sama',
    'udh': 'sudah', 'udah': 'sudah', 'sdh': 'sudah', 'dah': 'sudah',
    'blm': 'belum', 'belom': 'belum', 'blom': 'belum', 'msih': 'masih',
    'msh': 'masih', 'lg': 'lagi', 'lgi': 'lagi', 'dlu': 'dulu', 'dl': 'dulu',
    'mnt': 'menit', 'dtk': 'detik', 'smpe': 'sampai', 'sampe': 'sampai',
    'smpai': 'sampai', 'jg': 'juga', 'bs': 'bisa', 'bsa': 'bisa',
    'jt': 'juta', 'rb': 'ribu', 'aja': 'saja', 'aj': 'saja',
    'emng': 'memang', 'emang': 'memang', 'ad': 'ada',
    'gimana': 'bagaimana', 'gmn': 'bagaimana', 'bgimana': 'bagaimana',
    'kalo': 'kalau', 'nyoba': 'mencoba', 'gopey': 'gopay', 'bayr': 'bayar',
    'nasabh': 'nasabah', 'nadabah': 'nasabah', 'pembayran': 'pembayaran',
    'bgus': 'bagus', 'bgs': 'bagus', 'peroses': 'proses', 'sanggat': 'sangat',
    'trimakasih': 'terima kasih', 'app': 'aplikasi', 'apk': 'aplikasi',
    'tf': 'transfer', 'topup': 'isi ulang', 'okey': 'ok', 'okay': 'ok',
    'oke': 'ok', 'eror': 'error', 'err': 'error', 'loading': 'memuat',
    'load': 'memuat', 'update': 'perbarui', 'updt': 'perbarui',
    'duit': 'uang', 'duwit': 'uang', 'thanks': 'terima kasih',
    'thx': 'terima kasih', 'makasih': 'terima kasih', 'mksh': 'terima kasih',
    'tq': 'terima kasih', 'tlg': 'tolong', 'tlong': 'tolong',
    'lmbt': 'lambat', 'lemot': 'lambat',
}

KATA_DIPERTAHANKAN = {
    'tidak', 'bukan', 'belum', 'jangan', 'kurang', 'tak', 'tanpa',
    'enggak', 'baik',
}

FILLER_INFORMAL = {
    'sih', 'deh', 'loh', 'lah', 'dong', 'kan', 'nih', 'tuh', 'nya', 'ku',
    'mu', 'kayak', 'kaya', 'yah', 'wah', 'hah', 'eh', 'ah', 'oh', 'iya',
    'ya', 'hai', 'hay', 'hei',
}

DOMAIN_APP = {
    'gopay', 'gojek', 'google', 'play', 'store', 'android', 'hp', 'ok',
}

# ---------------------------------------------------------------------------
# Inisialisasi stopword & stemmer
# ---------------------------------------------------------------------------
_stopwords_default = set(StopWordRemoverFactory().get_stop_words())
_stopwords_default -= KATA_DIPERTAHANKAN
ALL_STOPWORDS = _stopwords_default | FILLER_INFORMAL | DOMAIN_APP

_stemmer = StemmerFactory().create_stemmer()

@lru_cache(maxsize=50_000)
def _stem(kata: str) -> str:
    return _stemmer.stem(kata)

# ---------------------------------------------------------------------------
# Fungsi Tahapan Preprocessing (Identik Eksperimen)
# ---------------------------------------------------------------------------
def langkah_case_folding(teks):
    return teks.lower()

def langkah_cleansing(teks):
    teks = re.sub(r"http\S+|www\.\S+", " ", teks)   # URL
    teks = re.sub(r"[@#]\w+", " ", teks)            # mention & hashtag
    teks = re.sub(r"\d+", " ", teks)                # angka
    teks = re.sub(r"[^a-z\s]", " ", teks)           # selain huruf jadi spasi
    return teks

def langkah_reduksi_karakter(teks):
    return re.sub(r"(.)\1{2,}", r"\1", teks)        # 'bagusss' -> 'bagus'

def langkah_normalisasi_spasi(teks):
    return re.sub(r"\s+", " ", teks).strip()

def langkah_tokenisasi(teks):
    try:
        return word_tokenize(teks)
    except Exception:
        return teks.split()                          # cadangan jika NLTK gagal

def langkah_normalisasi_slang(tokens):
    hasil = [KAMUS_SLANG.get(t, t) for t in tokens]
    return " ".join(hasil).split()                   # pecah lagi nilai multi-kata

def langkah_stopword(tokens):
    return [t for t in tokens if t not in ALL_STOPWORDS]

def langkah_stemming(tokens):
    return [_stem(t) for t in tokens]

# ---------------------------------------------------------------------------
# Main Preprocess Function
# ---------------------------------------------------------------------------
def preprocess(teks: str) -> str:
    """Kembalikan teks bersih siap-fitur, 100% identik dengan notebook 02."""
    teks = langkah_case_folding(str(teks))
    teks = langkah_cleansing(teks)
    teks = langkah_reduksi_karakter(teks)
    teks = langkah_normalisasi_spasi(teks)
    tok = langkah_tokenisasi(teks)
    tok = langkah_normalisasi_slang(tok)
    tok = langkah_stopword(tok)
    tok = langkah_stemming(tok)
    tok = langkah_stopword(tok)                      # Stopword kedua setelah stemming
    return " ".join(tok)
