"""
Tema visual dashboard.

Arah desain: editorial cetak yang hangat. Latar kertas, tinta gelap, satu
font tampilan berkarakter (Fraunces) dipasangkan dengan font teks yang bersih
(Spline Sans). Warna sentimen dipakai konsisten di seluruh grafik. Tujuannya
agar tampilan terasa dirancang, bukan tempelan template.
"""

import streamlit as st

COLORS = {
    "bg": "#f4efe4",
    "surface": "#fffdf7",
    "ink": "#1f1d18",
    "muted": "#6d675b",
    "line": "#e0d8c6",
}

# Warna per kelas sentimen, dipakai seragam di semua grafik
SENT = {
    "positif": "#2f7d5b",
    "negatif": "#bf513a",
    "netral": "#9a8f76",
}

SENT_LABEL = {
    "positif": "Positif",
    "negatif": "Negatif",
    "netral": "Netral",
}


def inject():
    """Suntikkan font dan gaya kustom ke halaman."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Spline+Sans:wght@400;500;600&display=swap');

        :root {
            --bg: #f4efe4;
            --surface: #fffdf7;
            --ink: #1f1d18;
            --muted: #6d675b;
            --line: #e0d8c6;
        }

        .stApp { background: var(--bg); }

        html, body, [class*="css"], .stMarkdown, p, span, label, div {
            font-family: 'Spline Sans', system-ui, sans-serif;
            color: var(--ink);
        }

        h1, h2, h3, h4 {
            font-family: 'Fraunces', Georgia, serif !important;
            color: var(--ink);
            letter-spacing: -0.01em;
        }

        /* Header sampul */
        .gp-cover {
            border-bottom: 2px solid var(--ink);
            padding: 0.4rem 0 1.1rem 0;
            margin-bottom: 1.4rem;
        }
        .gp-cover .eyebrow {
            font-family: 'Spline Sans', sans-serif;
            text-transform: uppercase;
            letter-spacing: 0.22em;
            font-size: 0.72rem;
            color: var(--muted);
            margin-bottom: 0.35rem;
        }
        .gp-cover h1 {
            font-size: 2.5rem;
            line-height: 1.05;
            margin: 0;
            font-weight: 600;
        }
        .gp-cover .lede {
            color: var(--muted);
            max-width: 42rem;
            margin-top: 0.6rem;
            font-size: 1rem;
        }

        /* Kartu metrik */
        .gp-cards { display: flex; gap: 0.9rem; flex-wrap: wrap; margin: 0.4rem 0 0.6rem 0; }
        .gp-card {
            flex: 1 1 160px;
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 2px;
            padding: 1rem 1.1rem;
        }
        .gp-card .k { font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.12em; color: var(--muted); }
        .gp-card .v { font-family: 'Fraunces', serif; font-size: 2rem; font-weight: 600; line-height: 1.1; margin-top: 0.2rem; }
        .gp-card .n { font-size: 0.8rem; color: var(--muted); margin-top: 0.15rem; }

        /* Lencana sentimen */
        .gp-badge {
            display: inline-block; padding: 0.25rem 0.7rem; border-radius: 999px;
            font-weight: 600; font-size: 0.95rem; color: #fff;
        }

        /* Keterangan kecil */
        .gp-note {
            font-size: 0.85rem; color: var(--muted);
            border-left: 2px solid var(--line); padding-left: 0.7rem; margin: 0.5rem 0;
        }

        /* Tab */
        .stTabs [data-baseweb="tab-list"] { gap: 1.4rem; border-bottom: 1px solid var(--line); }
        .stTabs [data-baseweb="tab"] {
            font-family: 'Fraunces', serif; font-size: 1.02rem; padding: 0.2rem 0;
            color: var(--muted);
        }
        .stTabs [aria-selected="true"] { color: var(--ink) !important; }

        /* Tombol */
        .stButton > button {
            background: var(--ink); color: var(--bg); border: none; border-radius: 2px;
            font-weight: 600; padding: 0.45rem 1.2rem;
        }
        .stButton > button:hover { background: #3a362d; color: var(--bg); }

        /* Chip kata */
        .gp-chip {
            display: inline-block; margin: 0.12rem; padding: 0.18rem 0.55rem;
            border: 1px solid var(--line); border-radius: 2px; background: var(--surface);
            font-size: 0.85rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def cover(eyebrow: str, judul: str, lede: str):
    st.markdown(
        f"""
        <div class="gp-cover">
          <div class="eyebrow">{eyebrow}</div>
          <h1>{judul}</h1>
          <div class="lede">{lede}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def cards(items):
    """items: list of (label, value, note)."""
    blok = "".join(
        f'<div class="gp-card"><div class="k">{k}</div>'
        f'<div class="v">{v}</div><div class="n">{n}</div></div>'
        for k, v, n in items
    )
    st.markdown(f'<div class="gp-cards">{blok}</div>', unsafe_allow_html=True)


def badge(kelas: str):
    warna = SENT.get(kelas, COLORS["muted"])
    teks = SENT_LABEL.get(kelas, kelas)
    st.markdown(
        f'<span class="gp-badge" style="background:{warna}">{teks}</span>',
        unsafe_allow_html=True,
    )


def note(teks: str):
    st.markdown(f'<div class="gp-note">{teks}</div>', unsafe_allow_html=True)


def chips(kata_list):
    blok = "".join(f'<span class="gp-chip">{k}</span>' for k in kata_list)
    st.markdown(blok, unsafe_allow_html=True)


def plotly_layout(fig):
    """Seragamkan tampilan grafik Plotly dengan tema."""
    fig.update_layout(
        font_family="Spline Sans, sans-serif",
        font_color=COLORS["ink"],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
    )
    fig.update_xaxes(gridcolor=COLORS["line"], zeroline=False)
    fig.update_yaxes(gridcolor=COLORS["line"], zeroline=False)
    return fig
