from pathlib import Path
import math
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Data Mining KNN", page_icon="📊", layout="wide")
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data_pasien.csv"
FEATURES = ["Usia", "BMI", "Glukosa", "Tekanan Darah", "Riwayat Keluarga"]

st.markdown("""
<style>
.stApp {background:#f4f7fb}
.main .block-container {padding-top:1.6rem;max-width:1250px}
.hero {padding:1.5rem 1.8rem;border-radius:18px;color:white;background:linear-gradient(120deg,#183153,#2457a7);margin-bottom:1rem}
.hero h1 {color:white;margin:0;font-size:2rem}
.hero p {color:#e2ecff;margin:.5rem 0 0}
div[data-testid="stMetric"] {background:white;border:1px solid #e3eaf4;padding:1rem;border-radius:14px}
</style>
<div class="hero">
<h1>📊 Data Mining — Algoritma KNN</h1>
<p>Aplikasi klasifikasi dengan metode K-Nearest Neighbors dan jarak Euclidean.</p>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError("data_pasien.csv tidak ditemukan. Letakkan file ini satu folder dengan app.py.")
    return pd.read_csv(DATA_FILE)

def calculate_knn(data, target, k):
    result = []
    for _, person in data.iterrows():
        squares = {f"{feature}²": (float(person[feature]) - float(target[feature])) ** 2 for feature in FEATURES}
        total = sum(squares.values())
        result.append({
            "Pasien": person["Pasien"], **squares,
            "Jumlah Kuadrat": total, "Jarak Euclidean": math.sqrt(total), "Kelas": person["Kelas"]
        })
    distances = pd.DataFrame(result).sort_values("Jarak Euclidean").reset_index(drop=True)
    distances.insert(0, "Peringkat", range(1, len(distances) + 1))
    nearest = distances.head(k).copy()
    votes = nearest["Kelas"].value_counts().to_dict()
    sums = nearest.groupby("Kelas")["Jarak Euclidean"].sum().to_dict()
    winner = sorted(votes, key=lambda cls: (-votes[cls], sums[cls]))[0]
    return distances, nearest, votes, winner

st.sidebar.header("⚙️ Input Pasien Baru")
st.sidebar.caption("Masukkan nilai untuk klasifikasi.")
with st.sidebar.form("input_knn"):
    usia = st.number_input("Usia", min_value=0, max_value=120, value=42, step=1)
    bmi = st.number_input("BMI", min_value=0.1, max_value=100.0, value=28.5, step=0.1)
    glukosa = st.number_input("Glukosa", min_value=0, max_value=1000, value=130, step=1)
    tekanan = st.number_input("Tekanan Darah", min_value=0, max_value=400, value=128, step=1)
    riwayat = st.selectbox("Riwayat Keluarga", [0, 1], format_func=lambda x: "0 — Tidak ada" if x == 0 else "1 — Ada")
    k = st.selectbox("Nilai K", [1, 3, 5, 7, 9], index=1)
    submitted = st.form_submit_button("🔎 Hitung KNN", type="primary", use_container_width=True)

try:
    data = load_data()
except Exception as e:
    st.error(f"Gagal memuat data: {e}")
    st.stop()

target = {"Usia": usia, "BMI": bmi, "Glukosa": glukosa, "Tekanan Darah": tekanan, "Riwayat Keluarga": riwayat}
if "knn_cache" not in st.session_state:
    st.session_state.knn_cache = calculate_knn(data, target, k)
    st.session_state.knn_target = target.copy()
    st.session_state.knn_k = k
if submitted:
    st.session_state.knn_cache = calculate_knn(data, target, k)
    st.session_state.knn_target = target.copy()
    st.session_state.knn_k = k

distances, nearest, votes, winner = st.session_state.knn_cache
used_target = st.session_state.knn_target
used_k = st.session_state.knn_k

st.subheader("Ringkasan Hasil")
c1, c2, c3 = st.columns(3)
c1.metric("Jumlah Data Latih", len(data))
c2.metric("Nilai K", used_k)
c3.metric("Hasil Klasifikasi", str(winner).upper())
st.success(f"**Kelas prediksi: {str(winner).upper()}** · Tetangga terdekat: " +
           ", ".join(f'{r["Pasien"]} ({r["Kelas"]})' for _, r in nearest.iterrows()))

left, right = st.columns(2)
with left:
    st.markdown("#### Data Pasien Baru")
    st.dataframe(pd.DataFrame([used_target]), use_container_width=True, hide_index=True)
with right:
    st.markdown("#### Jumlah Suara Kelas")
    st.dataframe(pd.DataFrame([{"Kelas": cls, "Jumlah suara": n} for cls, n in votes.items()]),
                 use_container_width=True, hide_index=True)

tab1, tab2, tab3, tab4 = st.tabs(["📋 Dataset Pasien", "📏 Semua Jarak", "🏆 Tetangga Terdekat", "🧮 Rumus"])
with tab1:
    st.dataframe(data, use_container_width=True, hide_index=True)
with tab2:
    display = distances[["Peringkat", "Pasien", "Jumlah Kuadrat", "Jarak Euclidean", "Kelas"]].copy()
    display["Jumlah Kuadrat"] = display["Jumlah Kuadrat"].map(lambda x: f"{x:.2f}")
    display["Jarak Euclidean"] = display["Jarak Euclidean"].map(lambda x: f"{x:.6f}")
    st.dataframe(display, use_container_width=True, hide_index=True)
with tab3:
    st.dataframe(nearest[["Peringkat", "Pasien", "Jarak Euclidean", "Kelas"]],
                 use_container_width=True, hide_index=True)
with tab4:
    st.latex(r"d(x,y)=\sqrt{\sum_{i=1}^{n}(x_i-y_i)^2}")
    st.markdown("1. Hitung selisih setiap fitur dan kuadratkan.\n2. Jumlahkan semua nilai kuadrat.\n3. Ambil akar kuadrat untuk mendapat jarak Euclidean.\n4. Ambil K tetangga terdekat dan pilih kelas dengan suara terbanyak.")

st.download_button("⬇️ Unduh hasil perhitungan CSV", distances.to_csv(index=False).encode("utf-8-sig"),
                   file_name="hasil_perhitungan_knn.csv", mime="text/csv")
st.caption("Catatan: proyek ini untuk pembelajaran algoritma, bukan alat diagnosis medis.")
