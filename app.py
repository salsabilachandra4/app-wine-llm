import time

import pandas as pd
import streamlit as st
from groq import Groq

from classifier import FITUR, classify, load_config

st.set_page_config(page_title="Klasifikasi Anggur (LLM)", page_icon="🤖")

# ---------- 1. Muat konfigurasi hasil evaluasi & siapkan client ----------
cfg = load_config("config.json")
rentang = cfg["rentang_fitur"]
median = cfg["median_fitur"]

try:
    API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:                       # file secrets belum ada atau key belum diisi
    st.error("GROQ_API_KEY belum diatur. Isi di .streamlit/secrets.toml (lokal) "
             "atau menu Secrets (Streamlit Community Cloud).")
    st.stop()


@st.cache_resource
def get_client():
    return Groq(api_key=API_KEY)


client = get_client()


@st.cache_data(show_spinner=False)
def prediksi(nilai: tuple):
    # hasil disimpan di cache: data yang sama tidak memanggil API dua kali (hemat kuota)
    return classify(client, cfg, dict(zip(FITUR, nilai)))


# ---------- 2. Sidebar: konfigurasi yang dikunci ----------
ev = cfg["evaluasi"]
with st.sidebar:
    st.header("Tentang Model")
    st.write(f"LLM: `{cfg['model']}` (Groq)")
    st.write(f"Prompt: {ev['konfigurasi']}")
    st.metric("Accuracy LLM (data uji)", f"{ev['accuracy']:.1%}")
    st.metric("F1 macro LLM (data uji)", f"{ev['f1_macro']:.3f}")
    if "f1_dnn_pembanding" in ev:
        st.metric("F1 macro DNN (pembanding)", f"{ev['f1_dnn_pembanding']:.3f}")
    st.caption(f"Dievaluasi pada {ev['n']} data uji yang sama dengan DNN, tanggal {ev['tanggal']}")
    if not ev.get("lolos_gerbang", True):
        st.warning(f"Konfigurasi ini **belum lolos** gerbang evaluasi "
                   f"(F1 macro ≥ {ev['target_f1']:.2f} dan invalid ≤ 5%).")
    with st.expander("Lihat system prompt"):
        st.code(cfg["system_prompt"], language=None)

st.title("🤖 Klasifikasi Kultivar Anggur dengan LLM")
st.write("Masukkan hasil analisis kimia satu sampel anggur. Data dikirim ke **Groq API**, lalu LLM "
         "menjawab kultivarnya: **Kultivar_1**, **Kultivar_2**, atau **Kultivar_3**.")

tab1, tab2 = st.tabs(["Prediksi Satu Sampel", "Prediksi dari File CSV"])

# ---------- 3. Prediksi satu data ----------
with tab1:
    st.caption("Nilai awal = median data latih.")
    with st.form("form_prediksi"):
        kolom = st.columns(3)
        nilai = {}
        for i, f in enumerate(FITUR):
            langkah = 1.0 if rentang[f][1] > 50 else 0.01
            nilai[f] = kolom[i % 3].number_input(f, min_value=0.0, value=float(median[f]),
                                                 step=langkah, format="%.0f" if langkah == 1 else "%.2f")
        kirim = st.form_submit_button("Prediksi")

    if kirim:
        luar = [c for c in FITUR if not rentang[c][0] <= nilai[c] <= rentang[c][1]]
        if luar:
            st.warning(f"Nilai {', '.join(luar)} di luar rentang data latih.")
        with st.spinner("Menghubungi Groq..."):
            hasil = prediksi(tuple(round(nilai[f], 2) for f in FITUR))
        if hasil in cfg["labels"]:
            st.success(f"Prediksi kultivar: **{hasil}**")
        else:
            st.info("Jawaban LLM di luar label (invalid) atau kuota habis. Coba lagi nanti.")

# ---------- 4. Prediksi banyak data dari CSV ----------
with tab2:
    st.write("Unggah CSV dengan kolom: " + ", ".join(f"`{c}`" for c in FITUR)
             + " (maksimal 30 baris agar hemat kuota).")
    with open("sampel_data_wine.csv", "rb") as f:
        st.download_button("⬇️ Unduh sampel data untuk dicoba", f,
                           file_name="sampel_data_wine.csv", mime="text/csv")
    file = st.file_uploader("Pilih file CSV", type="csv")
    if file is not None:
        df = pd.read_csv(file)
        kurang = [c for c in FITUR if c not in df.columns]
        if kurang:
            st.error(f"Kolom berikut tidak ditemukan: {', '.join(kurang)}")
        else:
            df = df.head(30)
            if st.button("Prediksi semua"):
                bar = st.progress(0.0)
                hasil = []
                for i, (_, b) in enumerate(df.iterrows(), start=1):
                    hasil.append(prediksi(tuple(round(float(b[f]), 2) for f in FITUR)))
                    bar.progress(i / len(df))
                    if i < len(df):
                        time.sleep(4)               # jaga batas request/token per menit
                df["prediksi_kultivar"] = hasil
                st.dataframe(df)
                st.write(df["prediksi_kultivar"].value_counts())
                if "kelas_asli" in df.columns:
                    benar = (df["prediksi_kultivar"] == df["kelas_asli"]).mean()
                    st.info(f"Kolom `kelas_asli` ditemukan: {benar:.0%} prediksi sesuai label asli.")
                st.download_button("Unduh hasil prediksi", df.to_csv(index=False),
                                   file_name="hasil_prediksi_llm.csv", mime="text/csv")
