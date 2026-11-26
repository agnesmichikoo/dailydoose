"""Streamlit app: prediksi kategori stres mahasiswa (prototipe PKM-KC)."""
import streamlit as st
from validation import validate_inputs
from predictor import predict_category, global_feature_importance

# Isi URL resmi layanan konseling kampus di sini. Dikosongkan sampai diberikan.
COUNSELING_URL = ""

st.set_page_config(page_title="Daily Doose", page_icon="🧠", layout="centered")

st.markdown("""
<style>
.block-container {max-width: 760px; padding-top: 1.5rem; padding-bottom: 3rem;}
#MainMenu, footer, header {visibility: hidden;}

/* Hero */
.hero {
  position: relative; overflow: hidden; padding: 2rem 1.75rem; border-radius: 22px; color: #fff;
  background: linear-gradient(135deg, #4F6BED 0%, #7B5CE6 55%, #B26BE8 100%);
  box-shadow: 0 12px 30px rgba(79,107,237,.28); margin-bottom: 1.25rem;
}
.hero::after {
  content: ""; position: absolute; right: -60px; top: -60px; width: 220px; height: 220px;
  border-radius: 50%; background: rgba(255,255,255,.12);
}
.hero .badge {
  display: inline-block; padding: .3rem .75rem; border-radius: 999px; background: rgba(255,255,255,.2);
  font-size: .8rem; letter-spacing: .02em; margin-bottom: .9rem;
}
.hero h1 {color: #fff; margin: 0 0 .5rem 0; font-size: 2rem; line-height: 1.2; font-weight: 800;}
.hero p {margin: 0; opacity: .95; font-size: 1.02rem; line-height: 1.6; max-width: 560px;}

/* Step cards */
.steps {display: grid; grid-template-columns: repeat(3, 1fr); gap: .9rem; margin: 1rem 0 1.25rem 0;}
.step {background: #fff; border-radius: 16px; padding: 1rem; box-shadow: 0 4px 14px rgba(20,25,50,.06); border: 1px solid #ECEEF5;}
.step .num {
  width: 30px; height: 30px; border-radius: 10px; display: flex; align-items: center; justify-content: center;
  background: #EEF1FD; color: #4F6BED; font-weight: 700; margin-bottom: .6rem;
}
.step h4 {margin: 0 0 .25rem 0; font-size: .98rem; color: #1F2433;}
.step p {margin: 0; font-size: .85rem; color: #5B6275; line-height: 1.45;}

/* Info & warning boxes */
.note {background: #EEF1FD; border-radius: 14px; padding: .9rem 1.1rem; color: #33407A; font-size: .95rem; margin: .5rem 0 1rem 0;}
.warn {
  background: #FFF6E5; border: 1px solid #F3C669; border-radius: 16px; padding: 1rem 1.2rem;
  color: #5A4300; font-size: .92rem; line-height: 1.55; margin-top: 1rem;
}
.warn strong {color: #7A5600;}

/* Result card */
.result {
  border-radius: 20px; padding: 1.5rem; color: #fff; margin: .5rem 0 1rem 0;
  box-shadow: 0 12px 26px rgba(20,25,50,.15);
}
.result .label {font-size: .85rem; opacity: .9; letter-spacing: .04em; text-transform: uppercase;}
.result .cat {font-size: 2.4rem; font-weight: 800; line-height: 1.1; margin: .25rem 0 .5rem 0;}
.result .desc {font-size: .98rem; line-height: 1.55; opacity: .97; margin: 0;}

/* Form section */
.section-title {font-weight: 700; color: #1F2433; font-size: 1.05rem; margin: .5rem 0 .25rem 0;}
.hint {color: #6B7286; font-size: .88rem; margin-bottom: .75rem;}

/* Buttons */
.stButton > button, .stFormSubmitButton > button, .stLinkButton > a {
  border-radius: 14px !important; font-weight: 600 !important; padding: .65rem 1rem !important;
}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
  box-shadow: 0 8px 18px rgba(79,107,237,.25);
}

/* Mobile */
@media (max-width: 640px) {
  .steps {grid-template-columns: 1fr;}
  .hero h1 {font-size: 1.6rem;}
  .result .cat {font-size: 2rem;}
}
</style>
""", unsafe_allow_html=True)

DISCLAIMER = (
    "Prediksi ini dihasilkan berdasarkan pola yang dipelajari dari Student Lifestyle Dataset dan mengikuti "
    "definisi kategori stres dalam dataset tersebut. Hasil ini bukan diagnosis psikologis, bukan pengukuran "
    "klinis tingkat stres, dan tidak menggantikan konsultasi dengan psikolog atau tenaga profesional. "
    "Apabila kamu merasa mengalami tekanan psikologis atau membutuhkan bantuan, silakan menghubungi layanan "
    "konseling kampus."
)

CATEGORY_INFO = {
    "Low": ("#2E9E6A", "#2E9E6A, #5BC995", "Kategori Low",
            "Menurut definisi dataset, Low berlaku ketika durasi belajar kurang dari 6 jam per hari dan durasi "
            "tidur minimal 6 jam per hari."),
    "Moderate": ("#D98E00", "#E0A100, #F4C542", "Kategori Moderate",
                 "Menurut definisi dataset, Moderate berlaku untuk kombinasi belajar dan tidur yang tidak termasuk "
                 "kategori Low maupun High."),
    "High": ("#D9534F", "#D9534F, #F07C6C", "Kategori High",
             "Menurut definisi dataset, High berlaku ketika durasi tidur kurang dari 6 jam per hari atau durasi "
             "belajar lebih dari 8 jam per hari."),
}

if "page" not in st.session_state:
    st.session_state.page = "home"
if "inputs" not in st.session_state:
    st.session_state.inputs = {}
if "result" not in st.session_state:
    st.session_state.result = None


def go(page):
    st.session_state.page = page


def show_home():
    st.markdown("""
    <div class="hero">
      <span class="badge">✨ Prototipe PKM-KC · AI & Psikologi Kognitif</span>
      <h1>Daily Doose</h1>
      <p style="font-weight:600; margin-bottom:.4rem;">Kenali pola harianmu</p>
      <p>Masukkan kebiasaan belajar, tidur, aktivitas fisik, dan aktivitas sosialmu. Aplikasi ini memprediksi
      kategori stres <b>Low, Moderate, atau High</b> menggunakan model Machine Learning.</p>
    </div>

    <div class="steps">
      <div class="step"><div class="num">1</div><h4>Isi 4 kebiasaan</h4><p>Dalam satuan jam per hari.</p></div>
      <div class="step"><div class="num">2</div><h4>Lihat prediksi</h4><p>Kategori dari model Decision Tree.</p></div>
      <div class="step"><div class="num">3</div><h4>Baca catatannya</h4><p>Hasil ini bukan diagnosis.</p></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="note">🔒 Aplikasi ini hanya untuk edukasi dan gambaran awal. '
        'Tidak menggantikan konsultasi dengan psikolog atau tenaga profesional.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    <div class="step" style="margin-bottom:1rem;">
      <h4>📏 Aturan pengisian</h4>
      <p style="margin-top:.4rem;">
        Model hanya dilatih dengan data dalam rentang berikut, sehingga jawaban di luar rentang
        <b>tidak diproses</b>:<br>
        • Belajar: <b>5–10 jam</b> per hari<br>
        • Tidur: <b>5–10 jam</b> per hari<br>
        • Aktivitas fisik: <b>0–13 jam</b> per hari<br>
        • Aktivitas sosial: <b>0–6 jam</b> per hari<br>
        • Total keempatnya tidak boleh lebih dari <b>24 jam</b>
      </p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Mulai pengisian  →", type="primary", use_container_width=True):
        go("input")
        st.rerun()


def show_input():
    st.markdown('<div class="section-title">Isi kebiasaan harianmu</div>', unsafe_allow_html=True)
    st.markdown('<div class="hint">Semua jawaban dalam <b>jam per hari</b>. Gunakan angka, misalnya 7 atau 6,5.</div>',
                unsafe_allow_html=True)

    with st.form("habits"):
        c1, c2 = st.columns(2)
        with c1:
            study = st.number_input("📚 Belajar (5–10 jam)", min_value=0.0, max_value=24.0, value=None,
                                    step=0.5, placeholder="Contoh: 7",
                                    help="Berapa jam kamu belajar dalam sehari? Rentang: 5 sampai 10 jam.")
            physical = st.number_input("🏃 Aktivitas fisik (0–13 jam)", min_value=0.0, max_value=24.0, value=None,
                                       step=0.5, placeholder="Contoh: 2",
                                       help="Berapa jam kamu melakukan aktivitas fisik? Rentang: 0 sampai 13 jam.")
        with c2:
            sleep = st.number_input("😴 Tidur (5–10 jam)", min_value=0.0, max_value=24.0, value=None,
                                    step=0.5, placeholder="Contoh: 7.5",
                                    help="Berapa jam kamu tidur dalam sehari? Rentang: 5 sampai 10 jam.")
            social = st.number_input("🗣️ Aktivitas sosial (0–6 jam)", min_value=0.0, max_value=24.0, value=None,
                                     step=0.5, placeholder="Contoh: 3",
                                     help="Berapa jam kamu melakukan aktivitas sosial? Rentang: 0 sampai 6 jam.")
        submitted = st.form_submit_button("Prediksi kategori  ✨", type="primary", use_container_width=True)

    if submitted:
        values = {"study": study, "sleep": sleep, "physical": physical, "social": social}
        errors = validate_inputs(values)
        if errors:
            st.session_state.result = None
            for e in errors:
                st.error(e)
            st.info(
                "Jika jam belajar, tidur, atau aktivitasmu memang berada di luar rentang ini, aplikasi tidak bisa "
                "memberikan prediksi. Kondisi seperti itu tetap layak dibicarakan dengan tenaga profesional atau "
                "layanan konseling kampus."
            )
        else:
            st.session_state.inputs = values
            st.session_state.result = predict_category(study, sleep, physical, social)
            go("result")
            st.rerun()

    if st.button("← Kembali ke beranda"):
        go("home")
        st.rerun()


def show_result():
    category = st.session_state.result
    v = st.session_state.inputs
    if category is None or not v:
        st.warning("Belum ada hasil. Silakan isi formulir terlebih dahulu.")
        if st.button("Ke formulir"):
            go("input")
            st.rerun()
        return

    _, grad, title, explanation = CATEGORY_INFO[category]
    st.markdown(
        f'<div class="result" style="background: linear-gradient(135deg, {grad});">'
        f'<div class="label">Hasil prediksi</div>'
        f'<div class="cat">{title}</div>'
        f'<p class="desc">{explanation}</p></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Masukan yang kamu berikan</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📚 Belajar", f'{v["study"]:g} jam')
    m2.metric("😴 Tidur", f'{v["sleep"]:g} jam')
    m3.metric("🏃 Fisik", f'{v["physical"]:g} jam')
    m4.metric("🗣️ Sosial", f'{v["social"]:g} jam')

    st.markdown('<div class="section-title" style="margin-top:1rem;">Kontribusi fitur secara umum</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="hint">Grafik ini menunjukkan kontribusi relatif tiap fitur pada model secara keseluruhan. '
        'Bukan alasan spesifik mengapa kamu mendapat kategori ini, dan bukan bukti hubungan sebab-akibat.</div>',
        unsafe_allow_html=True,
    )
    st.bar_chart(global_feature_importance(), color="#4F6BED")

    st.markdown(f'<div class="warn"><strong>⚠️ Peringatan</strong><br>{DISCLAIMER}</div>', unsafe_allow_html=True)
    st.write("")
    if COUNSELING_URL:
        st.link_button("💬 Hubungi layanan konseling kampus", COUNSELING_URL, use_container_width=True)
    else:
        st.info("[PLACEHOLDER: tautan layanan konseling kampus belum diisi. Isi COUNSELING_URL di app.py.]")

    c1, c2 = st.columns(2)
    if c1.button("🔄 Isi ulang", use_container_width=True):
        go("input")
        st.rerun()
    if c2.button("🏠 Beranda", use_container_width=True):
        go("home")
        st.rerun()


page = st.session_state.page
if page == "home":
    show_home()
elif page == "input":
    show_input()
else:
    show_result()
