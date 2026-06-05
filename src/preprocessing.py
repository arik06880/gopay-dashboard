"""
Preprocessing teks untuk dashboard.

Modul ini menyalin persis pipeline dari notebook 02 agar teks yang diketik
pengguna diproses dengan cara yang sama seperti data latih. Jika satu langkah
berbeda, prediksi akan menyimpang dari distribusi yang dipelajari model.

Urutan langkah (sesuai BAB 3 dan kode notebook 02):
  1. case folding
  2. hapus URL
  3. hapus mention & hashtag
  4. hapus angka
  5. ganti karakter non-huruf dengan spasi
  6. reduksi karakter berulang
  7. normalisasi spasi
  8. normalisasi slang
  9. tokenisasi (split berbasis spasi) + stopword removal
  10. stemming (Sastrawi)
  11. stopword removal kedua
"""

import re
from functools import lru_cache

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ---------------------------------------------------------------------------
# Kamus slang (101 entri) — disalin persis dari notebook 02
# ---------------------------------------------------------------------------
KAMUS_SLANG = {
    # Negasi
    'gk': 'tidak', 'ga': 'tidak', 'gak': 'tidak', 'ngga': 'tidak',
    'nggak': 'tidak', 'tdk': 'tidak', 'kaga': 'tidak', 'kagak': 'tidak',
    'ko': 'kok',
    # Intensitas
    'bgt': 'sangat', 'banget': 'sangat', 'bngt': 'sangat',
    # Kata ganti orang
    'sy': 'saya', 'aq': 'saya', 'gw': 'saya', 'gue': 'saya', 'gua': 'saya',
    'sya': 'saya', 'km': 'kamu', 'lo': 'kamu', 'lu': 'kamu',
    # Penghubung & preposisi
    'yg': 'yang', 'dg': 'dengan', 'dgn': 'dengan', 'dr': 'dari',
    'utk': 'untuk', 'tuk': 'untuk', 'krn': 'karena', 'karna': 'karena',
    'tp': 'tapi', 'tpi': 'tapi', 'pdhl': 'padahal', 'pdhal': 'padahal',
    'sm': 'sama',
    # Waktu & proses
    'udh': 'sudah', 'udah': 'sudah', 'sdh': 'sudah', 'dah': 'sudah',
    'blm': 'belum', 'belom': 'belum', 'blom': 'belum', 'msih': 'masih',
    'msh': 'masih', 'lg': 'lagi', 'lgi': 'lagi', 'dlu': 'dulu', 'dl': 'dulu',
    'mnt': 'menit', 'dtk': 'detik', 'smpe': 'sampai', 'sampe': 'sampai',
    'smpai': 'sampai',
    # Kata umum
    'jg': 'juga', 'bs': 'bisa', 'bsa': 'bisa', 'jt': 'juta', 'rb': 'ribu',
    'aja': 'saja', 'aj': 'saja', 'emng': 'memang', 'emang': 'memang',
    'ad': 'ada', 'gimana': 'bagaimana', 'gmn': 'bagaimana',
    'bgimana': 'bagaimana', 'kalo': 'kalau', 'nyoba': 'mencoba',
    # Typo umum dari data
    'gopey': 'gopay', 'bayr': 'bayar', 'nasabh': 'nasabah',
    'nadabah': 'nasabah', 'pembayran': 'pembayaran', 'bgus': 'bagus',
    'bgs': 'bagus', 'peroses': 'proses', 'sanggat': 'sangat',
    'trimakasih': 'terima kasih',
    # Aplikasi & transaksi
    'app': 'aplikasi', 'apk': 'aplikasi', 'tf': 'transfer',
    'topup': 'isi ulang',
    # Variasi 'ok'
    'okey': 'ok', 'okay': 'ok', 'oke': 'ok',
    # Teknis
    'eror': 'error', 'err': 'error', 'loading': 'memuat', 'load': 'memuat',
    'update': 'perbarui', 'updt': 'perbarui',
    # Keuangan
    'duit': 'uang', 'duwit': 'uang',
    # Apresiasi
    'thanks': 'terima kasih', 'thx': 'terima kasih', 'makasih': 'terima kasih',
    'mksh': 'terima kasih', 'tq': 'terima kasih',
    # Bantuan
    'tlg': 'tolong', 'tlong': 'tolong',
    # Performa
    'lmbt': 'lambat', 'lemot': 'lambat',
}

# Kata negasi & sentimen yang sengaja dipertahankan dari daftar stopword
KATA_DIPERTAHANKAN = {
    'tidak', 'bukan', 'belum', 'jangan', 'kurang', 'tak', 'tanpa',
    'enggak', 'baik',
}

# Filler informal — tidak mengandung sentimen
FILLER_INFORMAL = {
    'sih', 'deh', 'loh', 'lah', 'dong', 'kan', 'nih', 'tuh', 'nya', 'ku',
    'mu', 'kayak', 'kaya', 'yah', 'wah', 'hah', 'eh', 'ah', 'oh', 'iya',
    'ya', 'hai', 'hay', 'hei',
}

# Stopword domain aplikasi — terlalu umum, tidak diskriminatif
DOMAIN_APP = {
    'gopay', 'gojek', 'google', 'play', 'store', 'android', 'hp', 'ok',
}


# ---------------------------------------------------------------------------
# Inisialisasi stopword & stemmer (sekali, di tingkat modul)
# ---------------------------------------------------------------------------
_stopwords_default = set(StopWordRemoverFactory().get_stop_words())
_stopwords_default -= KATA_DIPERTAHANKAN
ALL_STOPWORDS = _stopwords_default | FILLER_INFORMAL | DOMAIN_APP

_stemmer = StemmerFactory().create_stemmer()


@lru_cache(maxsize=50_000)
def _stem(kata: str) -> str:
    return _stemmer.stem(kata)


def normalisasi_slang(teks: str) -> str:
    return ' '.join(KAMUS_SLANG.get(k, k) for k in teks.split())


def preprocess(teks: str) -> str:
    """Kembalikan teks bersih siap-fitur, identik dengan notebook 02."""
    if not isinstance(teks, str):
        teks = str(teks)
    teks = teks.lower()                                  # 1
    teks = re.sub(r'http\S+|www\.\S+', ' ', teks)        # 2
    teks = re.sub(r'[@#]\w+', ' ', teks)                 # 3
    teks = re.sub(r'\d+', ' ', teks)                     # 4
    teks = re.sub(r'[^a-z\s]', ' ', teks)                # 5
    teks = re.sub(r'(.)\1{2,}', r'\1', teks)             # 6
    teks = re.sub(r'\s+', ' ', teks).strip()             # 7
    teks = normalisasi_slang(teks)                       # 8
    kata = [k for k in teks.split() if k not in ALL_STOPWORDS]   # 9
    kata = [_stem(k) for k in kata]                              # 10
    kata = [k for k in kata if k not in ALL_STOPWORDS]           # 11
    return ' '.join(kata)
