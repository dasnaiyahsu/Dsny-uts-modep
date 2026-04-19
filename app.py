import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
 
# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Mono:wght@400;700&display=swap');
 
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
 
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        min-height: 100vh;
    }
 
    /* Header */
    .hero-header {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -1px;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        font-weight: 400;
        margin-top: 0;
    }
 
    /* Cards */
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        font-family: 'Space Mono', monospace;
        color: #a78bfa;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
        margin-top: 0.25rem;
    }
 
    /* Result Cards */
    .result-placed {
        background: linear-gradient(135deg, rgba(52, 211, 153, 0.15), rgba(16, 185, 129, 0.05));
        border: 1px solid rgba(52, 211, 153, 0.4);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
    }
    .result-not-placed {
        background: linear-gradient(135deg, rgba(248, 113, 113, 0.15), rgba(239, 68, 68, 0.05));
        border: 1px solid rgba(248, 113, 113, 0.4);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
    }
    .result-emoji {
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    .result-title {
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .result-placed .result-title { color: #34d399; }
    .result-not-placed .result-title { color: #f87171; }
    .result-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
    }
 
    /* Section headers */
    .section-header {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #a78bfa;
        margin-bottom: 0.75rem;
        margin-top: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(167, 139, 250, 0.2);
    }
 
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.8) !important;
        border-right: 1px solid rgba(255,255,255,0.08) !important;
    }
    [data-testid="stSidebar"] .stMarkdown p {
        color: #94a3b8;
    }
 
    /* Inputs */
    .stSlider [data-testid="stTickBar"] { display: none; }
    div[data-baseweb="select"] > div {
        background: rgba(255,255,255,0.05) !important;
        border-color: rgba(255,255,255,0.15) !important;
        color: white !important;
    }
    .stTextInput input, .stNumberInput input {
        background: rgba(255,255,255,0.05) !important;
        border-color: rgba(255,255,255,0.15) !important;
        color: white !important;
    }
 
    /* Fix semua label teks di sidebar supaya terlihat */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p {
        color: #e2e8f0 !important;
        font-weight: 500 !important;
    }
    [data-testid="stSidebar"] .stSlider [data-testid="stMarkdownContainer"] p {
        color: #e2e8f0 !important;
    }
    /* Slider value label */
    [data-testid="stSidebar"] [data-testid="stThumbValue"] {
        color: #a78bfa !important;
        font-weight: 700 !important;
    }
    /* Selectbox label */
    [data-testid="stSidebar"] .stSelectbox label {
        color: #e2e8f0 !important;
    }
    /* Number input label */
    [data-testid="stSidebar"] .stNumberInput label {
        color: #e2e8f0 !important;
    }
 
    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        width: 100% !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(124, 58, 237, 0.4) !important;
    }
 
    /* Divider */
    hr { border-color: rgba(255,255,255,0.08) !important; }
 
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
 
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #64748b;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(124, 58, 237, 0.3) !important;
        color: #a78bfa !important;
    }
</style>
""", unsafe_allow_html=True)
 
 
# ── Load Model ────────────────────────────────────────────────
@st.cache_resource
def load_model(path: str = "best_model.pkl"):
    if not Path(path).exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)
 
 
model = load_model()
 
 
# ── Hero Header ───────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-title">🎓 Placement Predictor</div>
    <p class="hero-subtitle">Prediksi peluang penempatan kerja mahasiswa berbasis Machine Learning</p>
</div>
""", unsafe_allow_html=True)
 
st.markdown("---")
 
# ── Sidebar — Input Form ──────────────────────────────────────
with st.sidebar:
    st.markdown("## ✏️ Input Data Mahasiswa")
    st.markdown("Isi data berikut untuk mendapatkan prediksi.")
 
    # ── Akademik ──────────────────────────────────────────────
    st.markdown('<div class="section-header">📚 Akademik</div>', unsafe_allow_html=True)
    cgpa              = st.slider("CGPA", 0.0, 10.0, 7.5, 0.1)
    tenth_percentage  = st.slider("Nilai SMP (%)", 40.0, 100.0, 75.0, 0.5)
    twelfth_percentage= st.slider("Nilai SMA (%)", 40.0, 100.0, 75.0, 0.5)
    backlogs          = st.slider("Jumlah Backlogs", 0, 20, 0)
    attendance_pct    = st.slider("Kehadiran (%)", 40.0, 100.0, 80.0, 0.5)
    study_hours       = st.slider("Jam Belajar/Hari", 0.0, 12.0, 4.0, 0.5)
 
    # ── Skill ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">💡 Keterampilan</div>', unsafe_allow_html=True)
    coding_skill   = st.slider("Coding Skill (1-10)", 1, 10, 6)
    comm_skill     = st.slider("Communication Skill (1-10)", 1, 10, 6)
    aptitude_skill = st.slider("Aptitude Skill (1-10)", 1, 10, 6)
 
    # ── Pengalaman ────────────────────────────────────────────
    st.markdown('<div class="section-header">🚀 Pengalaman</div>', unsafe_allow_html=True)
    projects      = st.slider("Proyek Selesai", 0, 20, 2)
    internships   = st.slider("Internship", 0, 5, 0)
    hackathons    = st.slider("Hackathon Diikuti", 0, 10, 1)
    certifications= st.slider("Sertifikasi", 0, 15, 1)
 
    # ── Gaya Hidup ────────────────────────────────────────────
    st.markdown('<div class="section-header">🌙 Gaya Hidup</div>', unsafe_allow_html=True)
    sleep_hours = st.slider("Jam Tidur/Hari", 3.0, 10.0, 7.0, 0.5)
    stress_level= st.slider("Tingkat Stres (1-10)", 1, 10, 5)
 
    # ── Profil ────────────────────────────────────────────────
    st.markdown('<div class="section-header">👤 Profil</div>', unsafe_allow_html=True)
    gender        = st.selectbox("Gender", ["Male", "Female"])
    branch        = st.selectbox("Jurusan", ["CSE", "ECE", "ME", "CE", "EE", "IT", "Other"])
    part_time_job = st.selectbox("Part-time Job", ["Yes", "No"])
    family_income = st.selectbox("Pendapatan Keluarga", ["Low", "Medium", "High"])
    city_tier     = st.selectbox("Tier Kota", ["Tier 1", "Tier 2", "Tier 3"])
    internet      = st.selectbox("Akses Internet", ["Yes", "No"])
    extracurricular = st.selectbox("Keterlibatan Ekstrakurikuler", ["Low", "Medium", "High", "Unknown"])
 
    st.markdown("---")
    predict_btn = st.button("🔮 Prediksi Sekarang", use_container_width=True)
 
 
# ── Main Content ──────────────────────────────────────────────
if model is None:
    st.error("⚠️ File `best_model.pkl` tidak ditemukan. Pastikan file berada di folder yang sama dengan `app.py`.")
    st.stop()
 
# Build input DataFrame
input_data = pd.DataFrame([{
    "cgpa":                      cgpa,
    "tenth_percentage":          tenth_percentage,
    "twelfth_percentage":        twelfth_percentage,
    "backlogs":                  backlogs,
    "attendance_percentage":     attendance_pct,
    "study_hours_per_day":       study_hours,
    "coding_skill_rating":       coding_skill,
    "communication_skill_rating":comm_skill,
    "aptitude_skill_rating":     aptitude_skill,
    "projects_completed":        projects,
    "internships_completed":     internships,
    "hackathons_participated":   hackathons,
    "certifications_count":      certifications,
    "sleep_hours":               sleep_hours,
    "stress_level":              stress_level,
    "gender":                    gender,
    "branch":                    branch,
    "part_time_job":             part_time_job,
    "family_income_level":       family_income,
    "city_tier":                 city_tier,
    "internet_access":           internet,
    "extracurricular_involvement": extracurricular,
}])
 
# Tabs
tab1, tab2, tab3 = st.tabs(["🔮 Prediksi", "📊 Profil Mahasiswa", "ℹ️ Tentang Model"])
 
# ══════════════════════════════════════════════════════════════
# TAB 1 — PREDIKSI
# ══════════════════════════════════════════════════════════════
with tab1:
    if predict_btn:
        prediction = model.predict(input_data)[0]
        probability= model.predict_proba(input_data)[0]
        placed_prob    = probability[1] * 100
        not_placed_prob= probability[0] * 100
 
        col1, col2 = st.columns([1.2, 1])
 
        with col1:
            if prediction == 1:
                st.markdown(f"""
                <div class="result-placed">
                    <div class="result-emoji">🎉</div>
                    <div class="result-title">PLACED!</div>
                    <p class="result-subtitle">Model memprediksi mahasiswa ini <strong>berhasil ditempatkan</strong> dengan tingkat keyakinan tinggi.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-not-placed">
                    <div class="result-emoji">📈</div>
                    <div class="result-title">Perlu Peningkatan</div>
                    <p class="result-subtitle">Model memprediksi mahasiswa ini <strong>belum berhasil ditempatkan</strong>. Masih ada waktu untuk meningkatkan skill!</p>
                </div>
                """, unsafe_allow_html=True)
 
            st.markdown("<br>", unsafe_allow_html=True)
 
            # Probability gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=placed_prob,
                title={"text": "Probabilitas Placed (%)", "font": {"color": "#94a3b8", "size": 14}},
                number={"font": {"color": "#a78bfa", "size": 36}, "suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#475569"},
                    "bar": {"color": "#7c3aed"},
                    "bgcolor": "rgba(255,255,255,0.05)",
                    "bordercolor": "rgba(255,255,255,0.1)",
                    "steps": [
                        {"range": [0, 40],  "color": "rgba(248,113,113,0.2)"},
                        {"range": [40, 70], "color": "rgba(251,191,36,0.2)"},
                        {"range": [70, 100],"color": "rgba(52,211,153,0.2)"},
                    ],
                    "threshold": {
                        "line": {"color": "#34d399", "width": 3},
                        "thickness": 0.75,
                        "value": 70
                    }
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=250,
                margin=dict(t=40, b=10, l=20, r=20),
                font={"family": "Plus Jakarta Sans"}
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
 
        with col2:
            # Probability bar chart
            fig_bar = go.Figure(go.Bar(
                x=["Not Placed", "Placed"],
                y=[not_placed_prob, placed_prob],
                marker_color=["#f87171", "#34d399"],
                marker_line_width=0,
                text=[f"{not_placed_prob:.1f}%", f"{placed_prob:.1f}%"],
                textposition="outside",
                textfont={"color": "white", "size": 14, "family": "Space Mono"},
            ))
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#94a3b8", "family": "Plus Jakarta Sans"},
                yaxis={"range": [0, 115], "gridcolor": "rgba(255,255,255,0.05)", "title": "Probabilitas (%)"},
                xaxis={"title": ""},
                height=280,
                margin=dict(t=20, b=10, l=20, r=20),
                title={"text": "Distribusi Probabilitas", "font": {"color": "#94a3b8", "size": 13}},
                showlegend=False,
            )
            st.plotly_chart(fig_bar, use_container_width=True)
 
            # Summary metrics
            st.markdown("**Ringkasan Input:**")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("CGPA", f"{cgpa:.1f}")
                st.metric("Coding Skill", f"{coding_skill}/10")
            with col_b:
                st.metric("Internship", f"{internships}x")
                st.metric("Proyek", f"{projects}x")
 
    else:
        # Placeholder state
        st.markdown("""
        <div style="text-align:center; padding: 4rem 2rem; color: #475569;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🎯</div>
            <div style="font-size: 1.2rem; font-weight: 600; color: #64748b;">Siap untuk Prediksi</div>
            <div style="font-size: 0.9rem; margin-top: 0.5rem;">Isi data mahasiswa di sidebar, lalu tekan tombol <strong style="color:#a78bfa">Prediksi Sekarang</strong></div>
        </div>
        """, unsafe_allow_html=True)
 
 
# ══════════════════════════════════════════════════════════════
# TAB 2 — PROFIL MAHASISWA (Radar Chart)
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Visualisasi Profil Mahasiswa")
    st.markdown("Radar chart menampilkan performa mahasiswa di berbagai dimensi dibanding rata-rata.")
 
    col1, col2 = st.columns(2)
 
    with col1:
        # Radar chart
        categories   = ["CGPA", "Coding", "Communication", "Aptitude", "Attendance", "Study Hours"]
        student_vals = [
            cgpa / 10 * 10,
            coding_skill,
            comm_skill,
            aptitude_skill,
            attendance_pct / 10,
            study_hours / 12 * 10,
        ]
        average_vals = [7.5, 6.0, 6.0, 6.0, 7.5, 4.5]
 
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=student_vals + [student_vals[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name="Mahasiswa",
            line_color="#a78bfa",
            fillcolor="rgba(167, 139, 250, 0.2)",
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=average_vals + [average_vals[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name="Rata-rata",
            line_color="#60a5fa",
            fillcolor="rgba(96, 165, 250, 0.1)",
            line_dash="dash",
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 10], gridcolor="rgba(255,255,255,0.1)", color="#64748b"),
                angularaxis=dict(color="#94a3b8"),
                bgcolor="rgba(0,0,0,0)",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#94a3b8", "family": "Plus Jakarta Sans"},
            legend={"font": {"color": "#94a3b8"}},
            height=380,
            margin=dict(t=30, b=30),
        )
        st.plotly_chart(fig_radar, use_container_width=True)
 
    with col2:
        # Metric cards
        st.markdown("**Ringkasan Profil:**")
        skill_avg = (coding_skill + comm_skill + aptitude_skill) / 3
        exp_score = internships * 2 + projects + hackathons + certifications
 
        metrics = [
            ("Rata-rata Skill",   f"{skill_avg:.1f}/10",  "💡"),
            ("Experience Score",  f"{exp_score} pts",      "🚀"),
            ("Kehadiran",         f"{attendance_pct:.0f}%","📅"),
            ("Jam Tidur/Hari",    f"{sleep_hours:.1f} jam","🌙"),
            ("Stres Level",       f"{stress_level}/10",    "😓"),
            ("Backlogs",          f"{backlogs} mata kuliah","📋"),
        ]
        for label, val, icon in metrics:
            st.markdown(f"""
            <div class="metric-card" style="margin-bottom:0.5rem; display:flex; align-items:center; justify-content:space-between; text-align:left; padding: 0.8rem 1.2rem;">
                <span style="color:#94a3b8; font-size:0.9rem;">{icon} {label}</span>
                <span style="color:#a78bfa; font-weight:700; font-family:'Space Mono',monospace;">{val}</span>
            </div>
            """, unsafe_allow_html=True)
 
        # Academic performance bar
        st.markdown("<br>**Performa Akademik:**", unsafe_allow_html=True)
        fig_acad = go.Figure(go.Bar(
            x=[cgpa * 10, tenth_percentage, twelfth_percentage],
            y=["CGPA (×10)", "Nilai SMP", "Nilai SMA"],
            orientation="h",
            marker_color=["#a78bfa", "#60a5fa", "#34d399"],
            text=[f"{cgpa:.1f}×10", f"{tenth_percentage:.0f}%", f"{twelfth_percentage:.0f}%"],
            textposition="outside",
            textfont={"color": "white", "size": 11},
        ))
        fig_acad.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#94a3b8", "family": "Plus Jakarta Sans"},
            xaxis={"range": [0, 115], "gridcolor": "rgba(255,255,255,0.05)"},
            yaxis={"color": "#94a3b8"},
            height=160,
            margin=dict(t=10, b=10, l=10, r=60),
            showlegend=False,
        )
        st.plotly_chart(fig_acad, use_container_width=True)
 
 
# ══════════════════════════════════════════════════════════════
# TAB 3 — TENTANG MODEL
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### ℹ️ Tentang Model")
 
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">RF</div>
            <div class="metric-label">Algoritma</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">0.939</div>
            <div class="metric-label">F1-Score</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">5K</div>
            <div class="metric-label">Data Training</div>
        </div>
        """, unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📋 Detail Model**")
        model_info = pd.DataFrame({
            "Atribut": ["Algoritma", "F1-Score", "Accuracy", "Precision", "Recall", "AUC-ROC", "Train:Test Split", "Cross-Validation"],
            "Nilai":   ["Random Forest", "0.9392", "0.8920", "0.9115", "0.9686", "0.8933", "80:20", "5-Fold Stratified"]
        })
        st.dataframe(model_info, hide_index=True, use_container_width=True)
 
    with col2:
        st.markdown("**📌 Fitur yang Digunakan**")
        fitur_info = pd.DataFrame({
            "Kategori":  ["Akademik", "Akademik", "Keterampilan", "Pengalaman", "Gaya Hidup", "Profil"],
            "Fitur":     ["CGPA, Nilai SMP/SMA, Backlogs", "Kehadiran, Jam Belajar",
                          "Coding, Komunikasi, Aptitude", "Internship, Proyek, Hackathon, Sertifikasi",
                          "Tidur, Stres", "Gender, Jurusan, Kota, Pendapatan"],
        })
        st.dataframe(fitur_info, hide_index=True, use_container_width=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("💡 Model dibangun menggunakan **scikit-learn Pipeline** dengan preprocessing otomatis (StandardScaler + OneHotEncoder) dan di-track menggunakan **MLflow**.")