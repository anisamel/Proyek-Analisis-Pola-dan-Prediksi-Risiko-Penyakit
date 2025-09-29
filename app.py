from ast import literal_eval
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import joblib
import json
from PIL import Image
from collections import Counter
from streamlit_option_menu import option_menu
from fpdf import FPDF
import tempfile
import base64


st.set_page_config(page_title="Prediksi Risiko Penyakit", layout="wide")
st.markdown("<h7 style='text-align: left;'>Website Prediksi Risiko Penyakit</h7>", unsafe_allow_html=True)

st.markdown("""
<style>
    /* Justify semua teks  */
    p {
        text-align: justify !important;
    }        
    /* Sidebar ukuran kecil */
    [data-testid="stSidebar"] {
        min-width: 280px;
        max-width: 280px;
        background-color: #000000;
    }
    /* sidebar disembunyikan */
    [data-testid="stSidebar"][aria-expanded="false"] {
        min-width: 0 !important;
        max-width: 0 !important;
        width: 0 !important;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    /* Container utama Streamlit */
    section.main > div {
        display: flex;
        justify-content: center;
        padding: 2rem;
    }
    /* Konten dibatasi dan dipusatkan */
    .block-container {
        max-width: 750px;
        margin: auto;
        width: 100%;
        padding-left: 2rem;
        padding-right: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# LOAD DATA 
@st.cache_data
def load_dataset():
    return pd.read_csv("dataset_preprocessing.csv")

@st.cache_data
def load_rules():
    return {
        "RH": pd.read_csv("rules_rh.csv"),
        "RD": pd.read_csv("rules_rd.csv"),
        "RJ": pd.read_csv("rules_rj.csv"),
        "RO": pd.read_csv("rules_ro.csv")
    }

@st.cache_resource
def load_models():
    return {
        "RH": joblib.load("model_rh.pkl"),
        "RD": joblib.load("model_rd.pkl"),
        "RJ": joblib.load("model_rj.pkl"),
        "RO": joblib.load("model_ro.pkl")
    }

@st.cache_resource
def load_feature_names():
    with open("label_columns.json") as f:
        return json.load(f)

df = load_dataset()
rules = load_rules()
models = load_models()
feature_names = load_feature_names()

# KAMUS DAN FUNGSI
user_to_alias = {
    'Usia': {'Remaja': 'A1', 'Dewasa': 'A2', 'Paruh Baya': 'A3', 'Lansia': 'A4'},
    'Jenis_Kelamin': {'Laki-laki': 'B1', 'Perempuan': 'B2'},
    'IMT': {'Normal': 'C1', 'Obesitas': 'C2', 'Overweight': 'C3', 'Underweight': 'C4'},
    'Riwayat_Penyakit_Keluarga': {'Diabetes': 'D1', 'Hipertensi': 'D2', 'Jantung': 'D3', 'Tidak ada': 'D4'},
    'Frekuensi_Sayur': {'Sangat Jarang': 'E1', 'Jarang': 'E2', 'Sedang': 'E3', 'Sering': 'E4'},
    'Frekuensi_Buah': {'Sangat Jarang': 'F1', 'Jarang': 'F2', 'Sedang': 'F3', 'Sering': 'F4'},
    'Frekuensi_Daging_Merah': {'Sangat Jarang': 'G1', 'Jarang': 'G2', 'Sedang': 'G3', 'Sering': 'G4'},
    'Frekuensi_Makanan_Cepat_Saji': {'Sangat Jarang': 'H1', 'Jarang': 'H2', 'Sedang': 'H3', 'Sering': 'H4'},
    'Frekuensi_Minuman_Bersoda': {'Sangat Jarang': 'I1', 'Jarang': 'I2', 'Sedang': 'I3', 'Sering': 'I4'},
    'Frekuensi_Alkohol': {'Sering': 'J1', 'Sesekali': 'J2', 'Tidak': 'J3'},
    'Frekuensi_Kafein': {'Kopi': 'K1', 'Minuman Energi': 'K2', 'Teh': 'K3', 'Tidak': 'K4'},
    'Aktivitas_Fisik': {'Banyak': 'L1', 'Sedang': 'L2', 'Sedikit': 'L3', 'Tidak Aktif': 'L4'},
    'Durasi_Tidur': {'Sangat Pendek': 'M1', 'Pendek': 'M2', 'Normal': 'M3'},
    'Tingkat_Stres': {'Rendah': 'N1', 'Sedang': 'N2', 'Tinggi': 'N3'},
    'Kebiasaan_Merokok': {'Sering': 'O1', 'Sesekali': 'O2', 'Tidak': 'O3'},
    'Work_Life_Balance': {'Baik': 'P1', 'Buruk': 'P2', 'Sedang': 'P3'},
    'Status_Ekonomi': {'Menengah': 'Q1', 'Rendah': 'Q2', 'Tinggi': 'Q3'},
    'Risiko_Penyakit': {'Risiko Hipertensi' : 'RH', 'Risiko Diabetes' : 'RD', 'Risiko Jantung' : 'RJ', 'Risiko Obesitas' : 'RO'}
}
rules_dict = {
    "RH": rules["RH"],
    "RD": rules["RD"],
    "RJ": rules["RJ"],
    "RO": rules["RO"]
}
risk_labels = {
    "RH": "Risiko Hipertensi",
    "RD": "Risiko Diabetes",
    "RJ": "Risiko Jantung",
    "RO": "Risiko Obesitas"
}
risk_label_to_code = {v: k for k, v in risk_labels.items()}

def kode_ke_deskripsi_dan_fitur(kode):
        for fitur, mapping in user_to_alias.items():
            for label, k in mapping.items():
                if k == kode.strip():
                    return label, fitur.replace('_', ' ')
        return kode, ''

#  SIDEBAR MENU 
menu_list = ["Beranda", "Informasi Dataset", "Aturan Asosiasi", "Prediksi Risiko"]
query_params = st.query_params
default_page = query_params.get("page", ["Beranda"])[0]
if default_page not in menu_list:
    default_page = "Beranda"

with st.sidebar:
    selected = option_menu(
        "Menu", 
        menu_list,
        icons=["house", "bar-chart", "search", "activity"],
        menu_icon="cast",
        default_index=menu_list.index(default_page),
        styles={
            "container": {
                "background-color": "#000000", 
                "padding": "0px",
                "margin": "0px",
                "box-shadow": "none",
                "border-radius": "0px"},
            "icon": {"color": "#FFFFFF", "font-size": "18px"}, 
            "nav-link": {
                "color": "#FFFFFF",
                "font-size": "16px",
                "text-align": "left",
                "margin": "2px",
                "border-radius": "6px"
            },
            "nav-link-selected": {
                "background-color": "rgba(255,255,255,0.2)", 
                "color": "#FFFFFF"
            }
        }
    )

if selected != default_page:
    st.query_params["page"] = selected

# HOMEPAGE 
if selected == "Beranda":
    st.markdown("<h1 style='text-align: center;'>Prediksi Risiko Penyakit</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; font-size: 18px;'>Selamat datang di aplikasi prediksi risiko penyakit berbasis machine learning. Sistem ini membantu Anda memantau dan memprediksi potensi risiko penyakit berdasarkan data pola makan dan gaya hidup Anda.</p>",
        unsafe_allow_html=True,
    )
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        image = Image.open("RO.png")
        st.image(image, use_container_width=True, caption="Risiko Obesitas")
    
    with col2:
        image = Image.open("RD.png")
        st.image(image, use_container_width=True, caption="Risiko Diabetes")
    with col3:
        image = Image.open("RH.png")
        st.image(image, use_container_width=True, caption="Risiko Hipertensi")
        
    with col4:
        image = Image.open("RJ2.png")
        st.image(image, use_container_width=True, caption="Risiko Jantung")
    

    st.markdown("""
    <div style='background-color:#000000; padding: 1rem 1.5rem; border-radius: 8px;'>
        <h3 color:#ffffff;'> Kenapa Perlu Memprediksi Risiko Penyakit? </h3>
        <p style='font-size:16px; color:#ffffff;'>
            Penyakit seperti <b>hipertensi</b>, <b>diabetes</b>, <b>obesitas</b>, dan <b>penyakit jantung</b> adalah penyebab utama masalah kesehatan masyarakat.
            Seringkali, penyakit-penyakit ini berkembang <b>tanpa gejala awal</b> dan baru terdeteksi saat sudah parah. Dengan memantau <b>pola makan</b>, <b>gaya hidup</b>, dan <b>faktor risiko</b> lainnya, Anda dapat melakukan <b>pencegahan sejak dini</b> dan menjaga kesehatan Anda secara proaktif.
        </p>
        <p style='font-size:13px; color:#ffffff;'> 
            Website ini membantu Anda mengenali potensi risiko berdasarkan data demografi, pola makan dan gaya hidup, sehingga Anda bisa mengambil langkah yang lebih sehat mulai dari sekarang.
        </p>
    
    </div>
    """, unsafe_allow_html=True)
        
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<h5 style='text-align: center;'> Info Dataset</h5>", unsafe_allow_html=True)
        st.markdown("- Lihat ringkasan data\n- Cek distribusi risiko\n- Cuplikan data yang digunakan")

    with col2:
        st.markdown("<h5 style='text-align: center;'> Aturan Asosiasi</h5>", unsafe_allow_html=True)
        st.markdown("- Lihat hasil asosiasi \n - Analisis keterkaitan pola")

    with col3:
        st.markdown("<h5 style='text-align: center;'> Prediksi Risiko</h5>", unsafe_allow_html=True)
        st.markdown("- Prediksi risiko penyakit dengan input data Anda\n - Lihat pola yang ditemukan\n- Lihat rekomendasi yang disarankan")


    st.markdown("---")
    st.markdown(
            "<p style='text-align: center; color: gray;'>Dikembangkan oleh Anisa Meilia | Informatika - Universitas Jenderal Soedirman</p>",
            unsafe_allow_html=True
    )

# INFORMASI DATASET 
elif selected == "Informasi Dataset":
    st.markdown("## Informasi Dataset")
    st.markdown(
        "<p style='text-align: justify;'>Dataset berisi data pola makan dan gaya hidup individu yang digunakan untuk analisis risiko terhadap beberapa jenis penyakit. Setiap kolom mewakili fitur yang telah dikategorikan dan dikodekan untuk proses machine learning. Pada halaman ini terdapat beberapa informasi mengenai dataset yang digunakan </p>",
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Jumlah Data", f"{df.shape[0]}")
    col2.metric("Jumlah Fitur", "60")
    col3.metric("Target", "4")

    st.subheader("Deskripsi Fitur dan Kode")
    fitur_data = {
        "Kode": [],
        "Fitur Asli": [],
        "Deskripsi": []
    }

    for fitur, mapping in user_to_alias.items():
        for k, v in mapping.items():
            fitur_data["Kode"].append(v)
            fitur_data["Fitur Asli"].append(fitur)
            fitur_data["Deskripsi"].append(k)

    fitur_df = pd.DataFrame(fitur_data)

    st.dataframe(
        fitur_df,
        use_container_width=True,
        height=400,
        hide_index=True
    )

    st.markdown("### Distribusi Risiko Penyakit")
    risk_cols = ['RH', 'RO', 'RJ', 'RD']
    risk_counts = df[risk_cols].sum()
    donut_fig = px.pie(
        names=risk_counts.index,
        values=risk_counts.values,
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    donut_fig.update_traces(textinfo='percent+label')
    st.plotly_chart(donut_fig, use_container_width=True)
    st.markdown("""
    <p style='font-size: 0.9rem; color: #dddddd;'>
    Distribusi jumlah risiko penyakit pada dataset belum seimbang. 
    Hal ini umum terjadi dan akan ditangani lebih lanjut untuk proses pemodelan algoritma.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("### Cuplikan Dataset")
    num_row = st.slider("Tampilkan baris:", min_value=5, max_value=100, value=10)
    st.dataframe(df.head(num_row), use_container_width=True)
    st.markdown("""
    <p style='font-size: 0.9rem; color: #dddddd;'>
    Setiap kolom dalam cuplikan dataset merupakan hasil <i>one-hot encoding</i>. 
    Nilai <b>1</b> menunjukkan fitur dimiliki oleh individu, sedangkan <b>0</b> berarti tidak dimiliki.
    Penjabaran arti kode fitur pada kolom cuplikan dataset dapat dilihat pada  tabel 'Deskripsi Fitur dan Kode'.
    </p>
    """, unsafe_allow_html=True)

    st.download_button("Unduh Dataset (CSV)", data=df.to_csv(index=False), file_name="dataset.csv", mime="text/csv")


# ASSOCIATION RULES 
elif selected == "Aturan Asosiasi":
    st.markdown("## Aturan Asosiasi Risiko Penyakit")
    st.markdown("Aturan asosiasi dapat digunakan untuk memahami pola yang sering muncul antara gejala dan risiko penyakit tertentu.")

    selected_label = st.selectbox("Pilih Risiko Penyakit", list(risk_label_to_code.keys()), index=0)
    risiko = risk_label_to_code[selected_label]
    st.info(f"Menampilkan aturan asosiasi untuk: **{selected_label}**")

    rules_df = rules[risiko]

    total_rules = len(rules_df)
    avg_conf = rules_df['confidence'].mean() if 'confidence' in rules_df.columns else 0
    avg_lift = rules_df['lift'].mean() if 'lift' in rules_df.columns else 0

    col1, col2, col3 = st.columns(3)
    col1.metric(" Total Aturan", total_rules)
    col2.metric(" Rata-rata Confidence", f"{avg_conf:.2f}")
    col3.metric("Rata-rata Lift", f"{avg_lift:.2f}")

    with st.expander(" Lihat Semua Aturan Asosiasi", expanded=False):
        selected_cols = ['antecedents', 'consequents', 'support', 'confidence', 'lift']
        cleaned_df = rules_df[selected_cols].copy()
        cleaned_df[['support', 'confidence', 'lift']] = cleaned_df[['support', 'confidence', 'lift']].round(1)
        
        st.dataframe(cleaned_df)

    st.subheader(" Visualisasi Persebaran Aturan Asosiasi Berdasarkan Support dan Confidence")

    scatter_fig = px.scatter(
        rules_df.sort_values(by='confidence', ascending=False).head(50),
        x='support',
        y='confidence',
        size='lift',
        hover_data=['antecedents', 'consequents'],
        color='lift',
        color_continuous_scale='Teal'
    )
    st.plotly_chart(scatter_fig, use_container_width=True)

    st.subheader("3 Aturan Asosiasi Terkuat")
    top_rules = rules_df.sort_values(by='confidence', ascending=False).head(3)

    for i, row in top_rules.iterrows():
        antecedent_kode = [x.strip() for x in row['antecedents'].split(',')]
        consequent_kode = [x.strip() for x in row['consequents'].split(',')]

        antecedent_pairs = [kode_ke_deskripsi_dan_fitur(k) for k in antecedent_kode]

        kode_target = consequent_kode[0]
        nama_target = risk_labels.get(kode_target, kode_target)
        antecedent_desc = [f"{fitur.lower()} {label}" for label, fitur in antecedent_pairs]

        alasan = ", ".join(antecedent_desc)

        st.markdown(f"""
        <div style='padding: 1rem; background-color: rgba(255,255,255,0.07); border-radius: 10px; margin-bottom: 10px;'>
            <b style='color:#ffffff;'>Rule {i+1}:</b> 
            <code style='color:#ffffff;'>{", ".join(antecedent_kode)} ➜ {", ".join(consequent_kode)}</code><br>
            <ul style='color:#ffffff;'>
                <li><b>Support:</b> {row['support']:.2f}</li>
                <li><b>Confidence:</b> {row['confidence']:.2f}</li>
                <li><b>Lift:</b> {row['lift']:.2f}</li>
            </ul>
            <p style='color:#ffffff; font-style: italic; margin-top: 8px;'>
                Individu dengan pola seperti <b>{alasan}</b> cenderung berpotensi <b>{nama_target}</b>.
            </p>
        </div>
        """, unsafe_allow_html=True)

# PREDIKSI RISIKO PENYAKIT
elif selected == "Prediksi Risiko":
    st.subheader("Masukkan Data Anda")

    # Kamus keterangan dropdown
    frekuensi_range = {
        'Sangat Jarang (0-1 kali/minggu)': 'Sangat Jarang',
        'Jarang (2 kali/minggu)': 'Jarang',
        'Sedang (3-4 kali/minggu)': 'Sedang',
        'Sering (5-6 kali/minggu)': 'Sering'
    }

    keterangan_dropdown = {
        'Frekuensi Konsumsi Sayur': frekuensi_range,
        'Frekuensi Konsumsi Buah': frekuensi_range,
        'Frekuensi Konsumsi Daging Merah': frekuensi_range,
        'Frekuensi Makanan Cepat Saji': frekuensi_range,
        'Frekuensi Minuman Bersoda': frekuensi_range,
        'Indeks Massa Tubuh (IMT)': {
            'Underweight (<18.5)': 'Underweight',
            'Normal (18.5 - 24.9)': 'Normal',
            'Overweight (25.0 - 27.0)': 'Overweight',
            'Obesitas (>=27.0)': 'Obesitas'
        },
        'Aktivitas Fisik': {
            'Tidak Aktif': 'Tidak Aktif',
            'Sedikit (1-2 kali/minggu)': 'Sedikit',
            'Sedang (3-5 kali/minggu)': 'Sedang',
            'Banyak (>5 kali/minggu)': 'Banyak'
        },
        'Durasi Tidur': {
            'Sangat Pendek (<4 jam)': 'Sangat Pendek',
            'Pendek (4-6 jam)': 'Pendek',
            'Normal (7-9 jam)': 'Normal'
        },
        'Status Ekonomi': {
            'Rendah (Pendapatan < 2 juta/bulan)': 'Rendah',
            'Menengah (Pendapatan 2-6 juta/bulan)': 'Menengah',
            'Tinggi (Pendepatan > 6 juta/bulan)': 'Tinggi'
        }
    }

    # Form Input
    col1, col2 = st.columns(2)
    with col1:
        usia = st.selectbox("Usia", ['Remaja', 'Dewasa', 'Paruh Baya', 'Lansia'])
        jenis_kelamin = st.selectbox("Jenis Kelamin", ['Laki-laki', 'Perempuan'])
        with st.expander("Hitung IMT Anda Disini", expanded=False):
            bb = st.number_input("Berat Badan (kg)", min_value=30, max_value=200, step=1)
            tb = st.number_input("Tinggi Badan (cm)", min_value=100, max_value=250, step=1)
            hitung_imt = st.button("Hitung IMT")

            imt_kategori = None
            if hitung_imt:
                imt_nilai = bb / ((tb/100) ** 2)
                if imt_nilai < 18.5:
                    imt_kategori = 'Underweight'
                elif imt_nilai < 25:
                    imt_kategori = 'Normal'
                elif imt_nilai < 27:
                    imt_kategori = 'Overweight'
                else:
                    imt_kategori = 'Obesitas'
                st.markdown(f"IMT Anda: {imt_nilai:.2f} -> Kategori: {imt_kategori}")
    
        imt = st.selectbox("Indeks Massa Tubuh (IMT)", list(keterangan_dropdown['Indeks Massa Tubuh (IMT)'].keys()))
        imt = keterangan_dropdown['Indeks Massa Tubuh (IMT)'][imt]

        riwayat_penyakit = st.selectbox("Riwayat Penyakit Keluarga", ['Tidak ada', 'Hipertensi', 'Diabetes', 'Jantung'])
        sayur = st.selectbox("Frekuensi Konsumsi Sayur", list(keterangan_dropdown['Frekuensi Konsumsi Sayur'].keys()))
        sayur = keterangan_dropdown['Frekuensi Konsumsi Sayur'][sayur]

        buah = st.selectbox("Frekuensi Konsumsi Buah", list(keterangan_dropdown['Frekuensi Konsumsi Buah'].keys()))
        buah = keterangan_dropdown['Frekuensi Konsumsi Buah'][buah]

        daging = st.selectbox("Frekuensi Konsumsi Daging Merah", list(keterangan_dropdown['Frekuensi Konsumsi Daging Merah'].keys()))
        daging = keterangan_dropdown['Frekuensi Konsumsi Daging Merah'][daging]

    with col2:
        fastfood = st.selectbox("Frekuensi Makanan Cepat Saji", list(keterangan_dropdown['Frekuensi Makanan Cepat Saji'].keys()))
        fastfood = keterangan_dropdown['Frekuensi Makanan Cepat Saji'][fastfood]

        soda = st.selectbox("Frekuensi Minuman Bersoda", list(keterangan_dropdown['Frekuensi Minuman Bersoda'].keys()))
        soda = keterangan_dropdown['Frekuensi Minuman Bersoda'][soda]

        alkohol = st.selectbox("Frekuensi Alkohol", ['Tidak', 'Sesekali', 'Sering'])
        kafein = st.selectbox("Konsumsi Kafein", ['Tidak', 'Kopi', 'Teh', 'Minuman Energi'])

        aktivitas = st.selectbox("Aktivitas Fisik", list(keterangan_dropdown['Aktivitas Fisik'].keys()))
        aktivitas = keterangan_dropdown['Aktivitas Fisik'][aktivitas]

        tidur = st.selectbox("Durasi Tidur", list(keterangan_dropdown['Durasi Tidur'].keys()))
        tidur = keterangan_dropdown['Durasi Tidur'][tidur]

        stres = st.selectbox("Tingkat Stres", ['Rendah', 'Sedang', 'Tinggi'])
        rokok = st.selectbox("Kebiasaan Merokok", ['Tidak', 'Sesekali', 'Sering'])
        wlb = st.selectbox("Work Life Balance", ['Baik', 'Sedang', 'Buruk'])
        ekonomi = st.selectbox("Status Ekonomi", list(keterangan_dropdown['Status Ekonomi'].keys()))
        ekonomi = keterangan_dropdown['Status Ekonomi'][ekonomi]


    if st.button("Prediksi Risiko Penyakit"):
        st.session_state['show_prediction'] = True

    if st.session_state.get('show_prediction', False):
        input_user = {
            'Usia': usia,
            'Jenis_Kelamin': jenis_kelamin,
            'IMT': imt,
            'Riwayat_Penyakit_Keluarga': riwayat_penyakit,
            'Frekuensi_Sayur': sayur,
            'Frekuensi_Buah': buah,
            'Frekuensi_Daging_Merah': daging,
            'Frekuensi_Makanan_Cepat_Saji': fastfood,
            'Frekuensi_Minuman_Bersoda': soda,
            'Frekuensi_Alkohol': alkohol,
            'Frekuensi_Kafein': kafein,
            'Aktivitas_Fisik': aktivitas,
            'Durasi_Tidur': tidur,
            'Tingkat_Stres': stres,
            'Kebiasaan_Merokok': rokok,
            'Work_Life_Balance': wlb,
            'Status_Ekonomi': ekonomi
        }

        kolom = feature_names  # hasil load dari label_columns.json
        encoded_input = pd.DataFrame([[0]*len(kolom)], columns=kolom)

        for key, val in input_user.items():
            kode = user_to_alias[key][val]
            if kode in encoded_input.columns:
                encoded_input.at[0, kode] = 1

        fitur_aktif = encoded_input.columns[encoded_input.iloc[0] == 1].tolist()
        
        #proses untuk save ke pdf
        def safe_text(text):
            # Hilangkan karakter yang bikin error di PDF
            return text.replace("→", "->")

        hasil_pdf = "Hasil Prediksi Risiko Penyakit\n\n"
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # Tambahkan Data Input Pengguna ke PDF
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Data Input Pengguna", ln=True)
        pdf.set_font("Arial", '', 12)
        for key, val in input_user.items():
            label = key.replace('_', ' ')
            pdf.cell(0, 8, f"{label}: {val}", ln=True)
        pdf.ln(5)
        pdf.set_draw_color(0, 0, 0)
        pdf.set_line_width(0.5)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)
        # Loop untuk Prediksi Setiap Risiko
        for risk in ['RH', 'RD', 'RJ', 'RO']:
            label_risk = risk_labels[risk]
            model = models[risk]
            pred = model.predict(encoded_input)[0]
            proba = model.predict_proba(encoded_input)[0][1]

            hasil_teks = f"{label_risk}\nPrediksi: {'Positif' if pred else 'Negatif'}\nProbabilitas: {proba*100:.2f}%\n"

            st.markdown(f"## {label_risk}")
            st.write(f"**Prediksi:** {'Positif' if pred else 'Negatif'}")
            st.write(f"**Probabilitas:** {proba*100:.2f}%")

            st.markdown(f"**Keterkaitan Pola Makan dan Gaya Hidup terhadap {label_risk} yang ditemukan dalam Input Anda:**")

            def kode_ke_fitur(kode):
                for fitur, mapping in user_to_alias.items():
                    for label, k in mapping.items():
                        if k == kode.strip():
                            fitur_bersih = fitur.replace('_', ' ')
                            return f"{k} ( {fitur_bersih} {label} )"
                return kode

            def ubah_itemset_fitur(itemset):
                items = [x.strip() for x in itemset.split(',') if x.strip()]
                return ', '.join([kode_ke_fitur(i) for i in items])

            matching_rules = []
            for _, row in rules_dict[risk].iterrows():
                try:
                    raw = row['antecedents']
                    antecedent = set([x.strip() for x in raw.split(',') if x.strip() != ''])
                    if antecedent.issubset(set(fitur_aktif)):
                        matching_rules.append(row)
                except:
                    continue

            if matching_rules:
                df_rules = pd.DataFrame(matching_rules).copy()
                df_rules['Antecedents (Sebab)'] = df_rules['antecedents'].apply(ubah_itemset_fitur)
                df_rules['Consequents (Akibat)'] = df_rules['consequents'].apply(lambda x: risk_labels.get(x.strip(), x.strip()))

                st.dataframe(
                    df_rules[['Antecedents (Sebab)', 'Consequents (Akibat)', 'confidence', 'lift']]
                    .sort_values(by='confidence', ascending=False)
                    .reset_index(drop=True)
                )

                hasil_teks += "\nKeterkaitan Pola Makan dan Gaya Hidup yang ditemukan dalam Input Anda:\n"
                for _, row in df_rules.iterrows():
                    hasil_teks += f"- Sebab: {row['Antecedents (Sebab)']} => Akibat: {row['Consequents (Akibat)']} (Conf: {row['confidence']:.2f}, Lift: {row['lift']:.2f})\n"

                st.markdown(f"""
                    <p style='font-size: 0.9rem; color: #dddddd;'>
                    Tabel ini menampilkan pola asosiasi atau kombinasi gaya hidup yang sering ditemukan dalam data pengguna lain pada dataset yang memiliki {label_risk}. 
                    Pola ini cocok dengan data input Anda saat ini sehingga digunakan sebagai dasar sistem dalam memberi rekomendasi.
                    </p>
                """, unsafe_allow_html=True)
            else:
                hasil_teks += "\nKeterkaitan Pola Makan dan Gaya Hidup yang ditemukan dalam Input Anda:\n"
                if pred == 1:
                    pesan = ("Tidak ditemukan pola gaya hidup serupa dalam aturan asosiasi (pola yang sering muncul pada dataset).")
                    st.info(pesan)
                    hasil_teks += f"{pesan}\n"
                else:
                    pesan = (f"Tidak ditemukan pola gaya hidup serupa dalam aturan asosiasi (pola yang sering muncul pada dataset)."
                            f" Hasil prediksi menunjukkan bahwa Anda "
                            f"tidak berisiko terhadap {risk_labels[risk]} saat ini. "
                            f" Tetap pertahankan kebiasaan sehat Anda, sedangkan untuk kebiasaan kurang sehat sebaiknya mulai diperbaiki.")
                    st.info(pesan)
                    hasil_teks += f"{pesan}\n"

            def get_fitur_asli(nama_fitur_ditampilkan):
                for fitur in user_to_alias.keys():
                    if fitur.replace('_', ' ') == nama_fitur_ditampilkan:
                        return fitur
                return nama_fitur_ditampilkan

            def interpret_rekomendasi_dari_rules(matching_rules, top_n=5):
                from collections import Counter

                semua_kode = []
                for rule in matching_rules:
                    antecedents = rule['antecedents'].split(',')
                    semua_kode.extend([a.strip() for a in antecedents if a.strip()])

                counter = Counter(semua_kode)
                rekom = []

                label_negatif = {
                    'frekuensi_sayur': ['Sangat Jarang', 'Jarang'],
                    'frekuensi_buah': ['Sangat Jarang', 'Jarang'],
                    'frekuensi_makanan_cepat_saji': ['Sering'],
                    'frekuensi_minuman_bersoda': ['Sering'],
                    'frekuensi_alkohol': ['Sering'],
                    'frekuensi_kafein': ['Minuman Energi', 'Kopi', 'Teh'],
                    'aktivitas_fisik': ['Tidak Aktif', 'Sedikit'],
                    'durasi_tidur': ['Sangat Pendek', 'Pendek'],
                    'tingkat_stres': ['Tinggi'],
                    'kebiasaan_merokok': ['Sering'],
                    'work_life_balance': ['Buruk']
                }

                fitur_baik_jika_tidak = ['frekuensi_alkohol', 'kebiasaan_merokok', 'frekuensi_kafein']

                def kode_ke_deskripsi_dan_fitur(kode):
                    for fitur, mapping in user_to_alias.items():
                        for label, k in mapping.items():
                            if k == kode.strip():
                                return label, fitur.replace('_', ' ')
                    return kode, kode

                for kode, jumlah in counter.most_common(top_n):
                    label, fitur_display = kode_ke_deskripsi_dan_fitur(kode)
                    fitur = get_fitur_asli(fitur_display)
                    fitur_bersih = fitur_display.lower()
                    fitur_key = fitur.lower()
                    label = label.strip()

                    if fitur_key in fitur_baik_jika_tidak and label == 'Tidak':
                        continue

                    if fitur_key in ['frekuensi_sayur', 'frekuensi_buah', 'aktivitas_fisik', 'durasi_tidur']:
                        rekom.append(f"Pertimbangkan untuk meningkatkan {fitur_bersih} Anda.")
                    elif fitur_key == 'work_life_balance' and label == 'Buruk':
                        rekom.append("Cobalah menyeimbangkan waktu kerja dan kehidupan pribadi Anda.")
                    elif fitur_key == 'tingkat_stres' and label == 'Tinggi':
                        rekom.append("Lakukan aktivitas relaksasi untuk membantu mengelola stres.")
                    elif fitur_key in label_negatif and label in label_negatif[fitur_key]:
                        rekom.append(f"Perbaiki atau kurangi {fitur_bersih} Anda, sebaiknya hindari kondisi '{label.lower()}' konsumsi.")
                    else:
                        rekom.append(f"Pertimbangkan untuk memperbaiki {fitur_bersih} Anda.")

                return rekom

            hasil_teks += "\n"
            if matching_rules:
                saran = interpret_rekomendasi_dari_rules(matching_rules)
                if saran:
                    if pred == 1:
                        st.markdown("**Rekomendasi Gaya Hidup Berdasarkan Pola Anda (Risiko Tinggi):**")
                        hasil_teks += "Rekomendasi Gaya Hidup Berdasarkan Pola Anda (Risiko Tinggi):\n"
                    else:
                        st.markdown("**Rekomendasi Gaya Hidup Berdasarkan Pola Anda (Risiko Rendah):**")
                        hasil_teks += "Rekomendasi Gaya Hidup Berdasarkan Pola Anda (Risiko Rendah):\n"

                    for s in saran:
                        st.markdown(f"- {s}")
                        hasil_teks += f"- {s}\n"
                else:
                    st.info("Tidak ada rekomendasi spesifik dari aturan.")
                    hasil_teks += "Tidak ada rekomendasi spesifik dari aturan.\n"
            else:
                if pred == 1:
                    pesan = f"Belum ada rekomendasi spesifik karena pola Anda tidak ditemukan dalam aturan asosiasi. Namun karena Anda terdeteksi berisiko terhadap {risk_labels[risk]}, Anda disarankan untuk meningkatkan kualitas pola makan, aktivitas fisik, dan kebiasaan gaya hidup sehat."
                    st.info(pesan)
                    hasil_teks += pesan + "\n"

            # Tambahkan header risiko dengan garis bawah sebagai pemisah
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, label_risk, ln=True)
            pdf.set_font("Arial", '', 12)
            pdf.multi_cell(0, 8, safe_text(hasil_teks))

            # Tambahkan garis horizontal sebagai pemisah antar risiko
            pdf.set_draw_color(0, 0, 0)
            pdf.set_line_width(0.5)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(5)  # spasi setelah garis

        # Simpan pdf ke temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            pdf.output(tmp_file.name)
            tmp_file_path = tmp_file.name

        # Baca file pdf sebagai bytes
        with open(tmp_file_path, "rb") as f:
            pdf_bytes = f.read()

        # Download button dengan bytes pdf
        st.download_button(
            label=" Download Hasil Prediksi sebagai PDF",
            data=pdf_bytes,
            file_name="hasil_prediksi_dan_rekomendasi.pdf",
            mime="application/pdf",
            help="Klik untuk mengunduh hasil prediksi risiko penyakit dalam format PDF"
        )