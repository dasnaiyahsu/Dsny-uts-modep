import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from pathlib import Path
 
st.set_page_config(
    page_title="GradTrack AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;700&display=swap');
* { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: #0a0a0f; min-height: 100vh; }
.stApp::before {
    content: '';
    position: fixed;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(ellipse at 20% 20%, rgba(124,58,237,0.12) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 80%, rgba(6,182,212,0.10) 0%, transparent 50%),
                radial-gradient(ellipse at 50% 50%, rgba(16,185,129,0.05) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d1a 0%, #0f0f1f 100%) !important;
    border-right: 1px solid rgba(124,58,237,0.2) !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label {
    color: #cbd5e1 !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
}
[data-testid="stSidebar"] [data-testid="stThumbValue"] {
    color: #a78bfa !important;
    font-weight: 700 !important;
}
div[data-baseweb="select"] > div {
    background: rgba(124,58,237,0.08) !important;
    border: 1px solid rgba(124,58,237,0.25) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}
div[data-baseweb="select"] svg { fill: #a78bfa !important; }
.stButton > button {
    background: linear-gradient(135deg, #7c3aed 0%, #06b6d4 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem 1.5rem !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    width: 100% !important;
    font-family: 'Inter', sans-serif !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 30px rgba(124,58,237,0.5) !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 12px !important;
    padding: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: #64748b !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(124,58,237,0.3), rgba(6,182,212,0.2)) !important;
    color: #a78bfa !important;
}
.glass-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
}
.result-placed {
    background: linear-gradient(135deg, rgba(16,185,129,0.1), rgba(6,182,212,0.05));
    border: 1px solid rgba(16,185,129,0.35);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    text-align: center;
}
.result-not-placed {
    background: linear-gradient(135deg, rgba(239,68,68,0.1), rgba(251,113,133,0.05));
    border: 1px solid rgba(239,68,68,0.35);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    text-align: center;
}
.sidebar-section {
    font-size: 0.7rem !important;
    font-weight: 800 !important;
    text-transform: uppercase !important;
    letter-spacing: 2.5px !important;
    color: #7c3aed !important;
    margin: 1.2rem 0 0.6rem !important;
    padding-bottom: 0.4rem !important;
    border-bottom: 1px solid rgba(124,58,237,0.2) !important;
}
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
    padding: 0.8rem !important;
}
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.75rem !important; }
[data-testid="stMetricValue"] { color: #a78bfa !important; font-size: 1.4rem !important; font-weight: 700 !important; }
hr { border-color: rgba(255,255,255,0.06) !important; margin: 1rem 0 !important; }
#MainMenu, footer, header { visibility: hidden; }
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-thumb { background: rgba(124,58,237,0.4); border-radius: 4px; }
@media (max-width: 768px) {
    .result-placed, .result-not-placed { padding: 1.5rem 1rem !important; }
}
</style>
""", unsafe_allow_html=True)
 
 
@st.cache_resource
def load_model(path="best_model.pkl"):
    if not Path(path).exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)
 
model = load_model()
 
# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 0.5rem 0 1rem;">
        <div style="font-size:2.5rem;">🎓</div>
        <div style="font-size:1.1rem; font-weight:800; color:#a78bfa;">GradTrack AI</div>
        <div style="font-size:0.72rem; color:#475569; margin-top:2px;">Placement Prediction System</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<hr>', unsafe_allow_html=True)
 
    st.markdown('<p class="sidebar-section">📚 Akademik</p>', unsafe_allow_html=True)
    cgpa               = st.slider("CGPA", 0.0, 10.0, 7.5, 0.1, format="%.1f")
    tenth_percentage   = st.slider("Nilai SMP (%)", 40.0, 100.0, 75.0, 0.5, format="%.1f")
    twelfth_percentage = st.slider("Nilai SMA (%)", 40.0, 100.0, 75.0, 0.5, format="%.1f")
    backlogs           = st.slider("Backlogs", 0, 20, 0)
    attendance_pct     = st.slider("Kehadiran (%)", 40.0, 100.0, 80.0, 0.5, format="%.1f")
    study_hours        = st.slider("Jam Belajar/Hari", 0.0, 12.0, 4.0, 0.5, format="%.1f")
 
    st.markdown('<p class="sidebar-section">💡 Keterampilan</p>', unsafe_allow_html=True)
    coding_skill   = st.slider("Coding Skill", 1, 10, 6)
    comm_skill     = st.slider("Communication Skill", 1, 10, 6)
    aptitude_skill = st.slider("Aptitude Skill", 1, 10, 6)
 
    st.markdown('<p class="sidebar-section">🚀 Pengalaman</p>', unsafe_allow_html=True)
    internships    = st.slider("Internship", 0, 5, 0)
    projects       = st.slider("Proyek Selesai", 0, 20, 2)
    hackathons     = st.slider("Hackathon Diikuti", 0, 10, 1)
    certifications = st.slider("Sertifikasi", 0, 15, 1)
 
    st.markdown('<p class="sidebar-section">🌙 Gaya Hidup</p>', unsafe_allow_html=True)
    sleep_hours  = st.slider("Jam Tidur/Hari", 3.0, 10.0, 7.0, 0.5, format="%.1f")
    stress_level = st.slider("Tingkat Stres (1-10)", 1, 10, 5)
 
    st.markdown('<p class="sidebar-section">👤 Profil</p>', unsafe_allow_html=True)
    gender          = st.selectbox("Gender", ["Male", "Female"])
    branch          = st.selectbox("Jurusan", ["CSE", "ECE", "ME", "CE", "EE", "IT", "Other"])
    part_time_job   = st.selectbox("Part-time Job", ["Yes", "No"])
    family_income   = st.selectbox("Pendapatan Keluarga", ["Low", "Medium", "High"])
    city_tier       = st.selectbox("Tier Kota", ["Tier 1", "Tier 2", "Tier 3"])
    internet        = st.selectbox("Akses Internet", ["Yes", "No"])
    extracurricular = st.selectbox("Ekstrakurikuler", ["Low", "Medium", "High", "Unknown"])
 
    st.markdown('<hr>', unsafe_allow_html=True)
    predict_btn = st.button("🔮 Prediksi Sekarang", use_container_width=True)
 
# ── HEADER ───────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 2rem 1rem 1rem;">
    <div style="
        font-size: clamp(1.8rem, 5vw, 3rem);
        font-weight: 900;
        background: linear-gradient(135deg, #a78bfa 0%, #06b6d4 50%, #10b981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -1.5px;
        line-height: 1.1;
        margin-bottom: 0.5rem;
    ">GradTrack AI</div>
    <p style="color:#64748b; font-size:clamp(0.85rem,2vw,1rem); font-weight:400;">
        Sistem prediksi penempatan kerja mahasiswa berbasis Machine Learning
    </p>
    <div style="display:inline-flex; gap:0.5rem; margin-top:0.8rem; flex-wrap:wrap; justify-content:center;">
        <span style="background:rgba(124,58,237,0.15);border:1px solid rgba(124,58,237,0.3);color:#a78bfa;padding:0.2rem 0.7rem;border-radius:999px;font-size:0.72rem;font-weight:600;">Random Forest</span>
        <span style="background:rgba(6,182,212,0.15);border:1px solid rgba(6,182,212,0.3);color:#22d3ee;padding:0.2rem 0.7rem;border-radius:999px;font-size:0.72rem;font-weight:600;">F1: 0.939</span>
        <span style="background:rgba(16,185,129,0.15);border:1px solid rgba(16,185,129,0.3);color:#34d399;padding:0.2rem 0.7rem;border-radius:999px;font-size:0.72rem;font-weight:600;">5.000 Data</span>
    </div>
</div>
""", unsafe_allow_html=True)
 
st.markdown("---")
 
if model is None:
    st.error("File `best_model.pkl` tidak ditemukan.")
    st.stop()
 
input_data = pd.DataFrame([{
    "cgpa": cgpa, "tenth_percentage": tenth_percentage,
    "twelfth_percentage": twelfth_percentage, "backlogs": backlogs,
    "attendance_percentage": attendance_pct, "study_hours_per_day": study_hours,
    "coding_skill_rating": coding_skill, "communication_skill_rating": comm_skill,
    "aptitude_skill_rating": aptitude_skill, "projects_completed": projects,
    "internships_completed": internships, "hackathons_participated": hackathons,
    "certifications_count": certifications, "sleep_hours": sleep_hours,
    "stress_level": stress_level, "gender": gender, "branch": branch,
    "part_time_job": part_time_job, "family_income_level": family_income,
    "city_tier": city_tier, "internet_access": internet,
    "extracurricular_involvement": extracurricular,
}])
 
tab1, tab2, tab3 = st.tabs(["🔮 Prediksi", "📊 Profil Mahasiswa", "ℹ️ Info Model"])
 
# ── TAB 1: PREDIKSI ───────────────────────────────────────────
with tab1:
    if predict_btn:
        prediction  = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        placed_prob     = probability[1] * 100
        not_placed_prob = probability[0] * 100
 
        col1, col2 = st.columns([1.1, 1], gap="medium")
 
        with col1:
            if prediction == 1:
                st.markdown(f"""
                <div class="result-placed">
                    <div style="font-size:3.5rem; margin-bottom:0.8rem;">🎉</div>
                    <div style="font-size:1.9rem; font-weight:900; color:#10b981; letter-spacing:-0.5px; margin-bottom:0.5rem;">PLACED!</div>
                    <p style="color:#94a3b8; font-size:0.9rem; line-height:1.6;">
                        Model memprediksi mahasiswa ini <strong style="color:#34d399">berhasil ditempatkan</strong>
                        dengan probabilitas <strong style="color:#34d399; font-family:'JetBrains Mono',monospace">{placed_prob:.1f}%</strong>
                    </p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-not-placed">
                    <div style="font-size:3.5rem; margin-bottom:0.8rem;">📈</div>
                    <div style="font-size:1.9rem; font-weight:900; color:#ef4444; letter-spacing:-0.5px; margin-bottom:0.5rem;">Perlu Peningkatan</div>
                    <p style="color:#94a3b8; font-size:0.9rem; line-height:1.6;">
                        Model memprediksi mahasiswa ini <strong style="color:#f87171">belum ditempatkan</strong>.
                        Probabilitas placed: <strong style="color:#f87171; font-family:'JetBrains Mono',monospace">{placed_prob:.1f}%</strong>
                    </p>
                </div>""", unsafe_allow_html=True)
 
            st.markdown("<br>", unsafe_allow_html=True)
 
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=placed_prob,
                title={"text": "Probabilitas Placed", "font": {"color": "#64748b", "size": 13}},
                number={"font": {"color": "#a78bfa", "size": 42, "family": "JetBrains Mono"}, "suffix": "%"},
                gauge={
                    "axis": {"range": [0,100], "tickcolor": "#334155", "tickwidth": 1, "tickfont": {"color": "#475569", "size": 10}},
                    "bar": {"color": "#7c3aed", "thickness": 0.25},
                    "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
                    "steps": [
                        {"range": [0, 40],  "color": "rgba(239,68,68,0.15)"},
                        {"range": [40, 70], "color": "rgba(251,191,36,0.15)"},
                        {"range": [70,100], "color": "rgba(16,185,129,0.15)"},
                    ],
                    "threshold": {"line": {"color": "#10b981", "width": 2}, "thickness": 0.8, "value": 70}
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=220, margin=dict(t=40, b=10, l=30, r=30),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
 
        with col2:
            fig_donut = go.Figure(go.Pie(
                values=[not_placed_prob, placed_prob],
                labels=["Not Placed", "Placed"],
                hole=0.65,
                marker_colors=["#ef4444", "#10b981"],
                marker_line=dict(color="#0a0a0f", width=3),
                textinfo="label+percent",
                textfont=dict(color="white", size=11),
                hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
            ))
            fig_donut.add_annotation(
                text=f"<b>{placed_prob:.0f}%</b><br>Placed",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="#a78bfa", family="JetBrains Mono"),
            )
            fig_donut.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                showlegend=True,
                legend=dict(font=dict(color="#94a3b8", size=11), orientation="h", x=0.15, y=-0.05),
                height=240, margin=dict(t=20, b=30, l=20, r=20),
                title=dict(text="Distribusi Probabilitas", font=dict(color="#64748b", size=12), x=0.5),
            )
            st.plotly_chart(fig_donut, use_container_width=True)
 
            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.metric("CGPA", f"{cgpa:.1f}/10")
                st.metric("Coding", f"{coding_skill}/10")
            with c2:
                st.metric("Internship", f"{internships}x")
                st.metric("Proyek", f"{projects}x")
    else:
        st.markdown("""
        <div style="text-align:center; padding:5rem 2rem;">
            <div style="font-size:5rem; margin-bottom:1rem; opacity:0.5;">🎯</div>
            <div style="font-size:1.1rem; font-weight:700; color:#475569; margin-bottom:0.5rem;">Siap Memprediksi</div>
            <div style="font-size:0.85rem; color:#334155; line-height:1.7;">
                Atur data mahasiswa di <strong style="color:#a78bfa">sidebar kiri</strong>,<br>
                lalu tekan <strong style="color:#06b6d4">Prediksi Sekarang</strong>
            </div>
        </div>""", unsafe_allow_html=True)
 
# ── TAB 2: PROFIL ─────────────────────────────────────────────
with tab2:
    st.markdown("### 📊 Visualisasi Profil Mahasiswa")
    col1, col2 = st.columns([1, 1], gap="medium")
 
    with col1:
        categories   = ["CGPA", "Coding", "Komunikasi", "Aptitude", "Kehadiran", "Belajar"]
        student_vals = [cgpa, coding_skill, comm_skill, aptitude_skill, attendance_pct/10, study_hours/12*10]
        avg_vals     = [7.5, 6.0, 6.0, 6.0, 7.5, 4.2]
 
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=student_vals+[student_vals[0]], theta=categories+[categories[0]],
            fill="toself", name="Kamu",
            line=dict(color="#a78bfa", width=2),
            fillcolor="rgba(124,58,237,0.15)",
            marker=dict(size=6, color="#a78bfa"),
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=avg_vals+[avg_vals[0]], theta=categories+[categories[0]],
            fill="toself", name="Rata-rata",
            line=dict(color="#06b6d4", width=1.5, dash="dot"),
            fillcolor="rgba(6,182,212,0.07)",
            marker=dict(size=4, color="#06b6d4"),
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0,10], gridcolor="rgba(255,255,255,0.06)", color="#475569", tickfont=dict(size=9)),
                angularaxis=dict(color="#64748b", tickfont=dict(size=10, color="#94a3b8")),
                bgcolor="rgba(0,0,0,0)",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#94a3b8"),
            legend=dict(font=dict(color="#94a3b8", size=11), orientation="h", x=0.2, y=-0.08),
            height=320, margin=dict(t=20, b=40),
            title=dict(text="Perbandingan Skill vs Rata-rata", font=dict(color="#64748b", size=12), x=0.5),
        )
        st.plotly_chart(fig_radar, use_container_width=True)
 
        fig_acad = go.Figure(go.Bar(
            x=[cgpa*10, tenth_percentage, twelfth_percentage],
            y=["CGPA x10", "Nilai SMP", "Nilai SMA"],
            orientation="h",
            marker=dict(color=["#a78bfa","#06b6d4","#10b981"], line=dict(width=0)),
            text=[f"{cgpa:.1f}x10", f"{tenth_percentage:.0f}%", f"{twelfth_percentage:.0f}%"],
            textposition="outside",
            textfont=dict(color="white", size=11, family="JetBrains Mono"),
        ))
        fig_acad.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#64748b"),
            xaxis=dict(range=[0,125], gridcolor="rgba(255,255,255,0.04)", showticklabels=False),
            yaxis=dict(color="#94a3b8", tickfont=dict(size=11)),
            height=150, margin=dict(t=30, b=10, l=10, r=70),
            showlegend=False,
            title=dict(text="Performa Akademik", font=dict(color="#64748b", size=12), x=0.5),
        )
        st.plotly_chart(fig_acad, use_container_width=True)
 
    with col2:
        exp_vals = [internships, projects, hackathons, certifications]
        exp_max  = [5, 20, 10, 15]
        exp_pct  = [v/m*100 for v,m in zip(exp_vals, exp_max)]
 
        fig_exp = go.Figure(go.Bar(
            x=exp_pct,
            y=["Internship", "Proyek", "Hackathon", "Sertifikasi"],
            orientation="h",
            marker=dict(color=["#a78bfa","#06b6d4","#10b981","#f59e0b"], line=dict(width=0)),
            text=[str(v) for v in exp_vals],
            textposition="outside",
            textfont=dict(color="white", size=11, family="JetBrains Mono"),
        ))
        fig_exp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#64748b"),
            xaxis=dict(range=[0,130], gridcolor="rgba(255,255,255,0.04)", showticklabels=False),
            yaxis=dict(color="#94a3b8", tickfont=dict(size=11)),
            height=200, margin=dict(t=30, b=10, l=10, r=60),
            showlegend=False,
            title=dict(text="Pengalaman (% dari maks)", font=dict(color="#64748b", size=12), x=0.5),
        )
        st.plotly_chart(fig_exp, use_container_width=True)
 
        wellness = max(0, min(100, sleep_hours/10*100 - stress_level*5))
        fig_well = go.Figure(go.Indicator(
            mode="gauge+number",
            value=wellness,
            title={"text": "Wellness Score", "font": {"color": "#64748b", "size": 12}},
            number={"font": {"color": "#10b981", "size": 28, "family": "JetBrains Mono"}, "suffix": "%"},
            gauge={
                "axis": {"range": [0,100], "tickcolor": "#334155", "tickfont": {"size": 9, "color": "#475569"}},
                "bar": {"color": "#10b981", "thickness": 0.25},
                "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
                "steps": [
                    {"range": [0, 40],  "color": "rgba(239,68,68,0.15)"},
                    {"range": [40, 70], "color": "rgba(251,191,36,0.12)"},
                    {"range": [70,100], "color": "rgba(16,185,129,0.15)"},
                ],
            }
        ))
        fig_well.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=180, margin=dict(t=30, b=10, l=20, r=20),
        )
        st.plotly_chart(fig_well, use_container_width=True)
 
        skill_avg = (coding_skill + comm_skill + aptitude_skill) / 3
        exp_score = internships*2 + projects + hackathons + certifications
        st.markdown(f"""
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.6rem; margin-top:0.5rem;">
            <div class="glass-card" style="padding:0.8rem; text-align:center;">
                <div style="font-size:1.3rem; font-weight:800; color:#a78bfa; font-family:'JetBrains Mono',monospace;">{skill_avg:.1f}</div>
                <div style="font-size:0.7rem; color:#475569; text-transform:uppercase; letter-spacing:1px; margin-top:2px;">Avg Skill</div>
            </div>
            <div class="glass-card" style="padding:0.8rem; text-align:center;">
                <div style="font-size:1.3rem; font-weight:800; color:#06b6d4; font-family:'JetBrains Mono',monospace;">{exp_score}</div>
                <div style="font-size:0.7rem; color:#475569; text-transform:uppercase; letter-spacing:1px; margin-top:2px;">Exp Score</div>
            </div>
            <div class="glass-card" style="padding:0.8rem; text-align:center;">
                <div style="font-size:1.3rem; font-weight:800; color:#10b981; font-family:'JetBrains Mono',monospace;">{attendance_pct:.0f}%</div>
                <div style="font-size:0.7rem; color:#475569; text-transform:uppercase; letter-spacing:1px; margin-top:2px;">Kehadiran</div>
            </div>
            <div class="glass-card" style="padding:0.8rem; text-align:center;">
                <div style="font-size:1.3rem; font-weight:800; color:#f59e0b; font-family:'JetBrains Mono',monospace;">{backlogs}</div>
                <div style="font-size:0.7rem; color:#475569; text-transform:uppercase; letter-spacing:1px; margin-top:2px;">Backlogs</div>
            </div>
        </div>""", unsafe_allow_html=True)
 
# ── TAB 3: INFO MODEL ─────────────────────────────────────────
with tab3:
    st.markdown("### Info Model")
    c1, c2, c3, c4 = st.columns(4)
    for col, (label, val, color) in zip(
        [c1,c2,c3,c4],
        [("F1-Score","0.939","#a78bfa"),("Accuracy","89.2%","#06b6d4"),("AUC-ROC","0.893","#10b981"),("Precision","91.2%","#f59e0b")]
    ):
        with col:
            st.markdown(f"""
            <div class="glass-card" style="text-align:center; padding:1.2rem 0.8rem;">
                <div style="font-size:1.6rem; font-weight:900; color:{color}; font-family:'JetBrains Mono',monospace;">{val}</div>
                <div style="font-size:0.7rem; color:#475569; text-transform:uppercase; letter-spacing:1.5px; margin-top:4px;">{label}</div>
            </div>""", unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown("**Detail Model**")
        st.dataframe(pd.DataFrame({
            "Atribut": ["Algoritma","n_estimators","max_depth","class_weight","Train:Test","CV"],
            "Nilai":   ["Random Forest","100","12","balanced","80:20","5-Fold Stratified"]
        }), hide_index=True, use_container_width=True)
    with col2:
        st.markdown("**Fitur Input**")
        st.dataframe(pd.DataFrame({
            "Kategori": ["Akademik","Keterampilan","Pengalaman","Gaya Hidup","Profil"],
            "Jumlah":   [6, 3, 4, 2, 7],
            "Contoh":   ["CGPA, Backlogs","Coding, Aptitude","Internship, Proyek","Tidur, Stres","Gender, Jurusan"]
        }), hide_index=True, use_container_width=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    models_list = ["Logistic\nRegression","Random\nForest","Gradient\nBoosting","SVM"]
    f1_scores   = [0.8892, 0.9392, 0.9364, 0.8948]
    auc_vals    = [0.9105, 0.8933, 0.9022, 0.8889]
 
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        name="F1-Score", x=models_list, y=f1_scores,
        marker_color=["#475569","#a78bfa","#7c3aed","#475569"],
        marker_line=dict(width=0),
        text=[f"{v:.4f}" for v in f1_scores],
        textposition="outside",
        textfont=dict(color="white", size=10, family="JetBrains Mono"),
    ))
    fig_comp.add_trace(go.Scatter(
        name="AUC-ROC", x=models_list, y=auc_vals,
        mode="lines+markers",
        line=dict(color="#06b6d4", width=2, dash="dot"),
        marker=dict(size=8, color="#06b6d4"),
        yaxis="y2",
    ))
    fig_comp.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#64748b", family="Inter"),
        yaxis =dict(range=[0.8,1.0], gridcolor="rgba(255,255,255,0.05)", title="F1-Score", color="#94a3b8"),
        yaxis2=dict(range=[0.8,1.0], overlaying="y", side="right", title="AUC-ROC", color="#06b6d4", showgrid=False),
        xaxis =dict(color="#94a3b8"),
        legend=dict(font=dict(color="#94a3b8"), orientation="h", x=0.3, y=1.12),
        height=280, margin=dict(t=40, b=20, l=20, r=60),
        title=dict(text="Perbandingan Semua Model", font=dict(color="#64748b", size=13), x=0.5),
    )
    st.plotly_chart(fig_comp, use_container_width=True)
    st.info("Model dibangun dengan scikit-learn Pipeline (StandardScaler + OneHotEncoder) dan di-track menggunakan MLflow. Model terbaik dipilih berdasarkan F1-Score tertinggi dari 4 algoritma.")