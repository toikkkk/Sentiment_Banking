import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import time

# KONFIGURASI HALAMAN

st.set_page_config(
    page_title="Dashboard Sentimen M-Banking",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)


# PROFESSIONAL COLOR PALETTE

COLORS = {
    'primary': '#2C5282',      # Navy Blue (Professional)
    'secondary': '#4A5568',    # Slate Gray
    'accent1': '#38A169',      # Muted Green
    'accent2': '#D69E2E',      # Muted Gold
    'accent3': '#3182CE',      # Light Blue
    'dark': '#1A202C',         # Charcoal
    'darker': '#0D1117',       # Very Dark
    'light': '#F7FAFC',        # Light Gray
    'white': '#FFFFFF'
}


# CUSTOM CSS - PROFESSIONAL MINIMALIST

st.markdown("""
<style>
    /* Main Background - Clean Dark */
    .stApp {
        background: linear-gradient(135deg, #0D1117 0%, #1A202C 100%);
    }
    
    /* Sidebar - Minimalist */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1A202C 0%, #0D1117 100%);
        border-right: 1px solid #2C5282;
    }
    
    /* Metric Cards - Professional */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #2C5282 0%, #4A5568 100%);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #2C5282;
        box-shadow: 0 4px 15px rgba(44, 82, 130, 0.3);
    }
    
    [data-testid="stMetricValue"] {
        color: #FFFFFF;
        font-size: 2rem;
        font-weight: 600;
    }
    
    [data-testid="stMetricLabel"] {
        color: #E2E8F0;
        font-size: 0.9rem;
    }
    
    /* Tabs - Clean */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #2C5282;
        border-radius: 8px;
        padding: 10px 20px;
        color: #FFFFFF;
        font-weight: 500;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background: #38A169;
        box-shadow: 0 2px 10px rgba(56, 161, 105, 0.3);
    }
    
    /* Header - Professional */
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 10px;
        letter-spacing: -0.5px;
    }
    
    .main-header-subtitle {
        font-size: 1.2rem;
        color: #94A3B8;
        text-align: center;
        margin-bottom: 20px;
        font-weight: 400;
    }
    
    /* Subheader */
    .sub-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #FFFFFF;
        margin-bottom: 15px;
        padding-bottom: 10px;
        border-bottom: 2px solid #2C5282;
    }
    
    /* Input Fields - Minimalist */
    [data-testid="stSidebar"] .stMultiSelect > div > div {
        background: #1A202C;
        border: 1px solid #2C5282;
        border-radius: 8px;
        color: #FFFFFF;
    }
    
    [data-testid="stSidebar"] .stDateInput > div > div {
        background: #1A202C;
        border: 1px solid #2C5282;
        border-radius: 8px;
        color: #FFFFFF;
    }
    
    /* Sidebar Text */
    [data-testid="stSidebar"] p {
        color: #E2E8F0;
        font-size: 0.85rem;
        font-weight: 500;
        margin-bottom: 8px;
    }
    
    /* Sidebar Divider */
    [data-testid="stSidebar"] hr {
        border-color: rgba(44, 82, 130, 0.4);
        margin: 15px 0;
    }
    
    /* Button - Professional */
    [data-testid="stSidebar"] .stButton > button {
        background: #2C5282;
        color: white;
        border: 1px solid #2C5282;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
        font-size: 0.9rem;
        transition: all 0.2s ease;
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #3182CE;
        border-color: #3182CE;
    }
    
    /* Dropdown Options */
    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {
        background: #2C5282;
        color: white;
        border-radius: 5px;
        font-size: 0.85rem;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #1A202C;
        border-radius: 8px;
        border: 1px solid #2C5282;
        color: #FFFFFF;
    }
    
    /* Text Color */
    p, label, div {
        color: #E2E8F0;
    }
    
    /* Dataframe */
    [data-testid="stDataFrame"] {
        background: #1A202C;
        border-radius: 10px;
        border: 1px solid #2C5282;
    }
    
    /* Remove Streamlit decorations */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Plotly Charts */
    .js-plotly-plot {
        border-radius: 10px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import time

# SESSION STATE UNTUK INTRO
if 'show_intro' not in st.session_state:
    st.session_state.show_intro = True

# HALAMAN INTRO - MINIMALIS & PROFESIONAL
if st.session_state.show_intro:
    # Hide sidebar dan elemen lain
    st.markdown("""
    <style>
    [data-testid="stSidebar"] {display: none;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stAlert {display: none;}
    
    /* Animasi fade in */
    @keyframes fadeInUp {
        from { 
            opacity: 0; 
            transform: translateY(30px); 
        }
        to { 
            opacity: 1; 
            transform: translateY(0); 
        }
    }
    
    .intro-container {
        animation: fadeInUp 1s ease-out;
        text-align: center;
    }
    
    .title-container {
        animation: fadeInUp 1s ease-out 0.3s both;
    }
    
    .subtitle-container {
        animation: fadeInUp 1s ease-out 0.5s both;
    }
    
    .progress-container {
        animation: fadeInUp 1s ease-out 0.7s both;
        max-width: 600px;
        margin: 40px auto;
    }
    
    .button-container {
        animation: fadeInUp 1s ease-out 0.9s both;
        max-width: 400px;
        margin: 30px auto;
    }
    
    /* Judul utama - SATU BARIS, RATA TENGAH */
    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        color: #FFFFFF;
        text-align: center;
        margin: 40px 0 20px 0;
        letter-spacing: -0.5px;
        line-height: 1.2;
    }
    
    .subtitle {
        font-size: 1.3rem;
        color: #94A3B8;
        text-align: center;
        margin: 15px 0;
        font-weight: 400;
    }
    
    .tech-stack {
        font-size: 0.95rem;
        color: #64748B;
        text-align: center;
        margin: 10px 0 50px 0;
        font-weight: 400;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #2C5282 0%, #4A5568 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 15px 40px;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #4A5568 0%, #2C5282 100%);
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(44, 82, 130, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Layout dengan columns untuk centering sempurna
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="intro-container">', unsafe_allow_html=True)
        
        # Logo/icon (SVG sederhana)
        st.markdown("""
        <div style="text-align: center; margin: 30px 0;">
            <svg width="80" height="80" viewBox="0 0 24 24" fill="none" 
                 xmlns="http://www.w3.org/2000/svg" style="margin: 0 auto; display: block;">
                <path d="M3 21h18v-8H3v8zm0-10h18V7H3v4zm0-6v4h18V5H3z" 
                      fill="#2C5282" opacity="0.8"/>
            </svg>
        </div>
        """, unsafe_allow_html=True)
        
        # Judul - SATU BARIS RATA TENGAH
        st.markdown("""
        <div class="title-container">
            <h1 class="main-title">CUSTOMER SENTIMENT DASHBOARD</h1>
        </div>
        """, unsafe_allow_html=True)
        
        # Subtitle
        st.markdown("""
        <div class="subtitle-container">
            <p class="subtitle">Analisis Sentimen Mobile Banking Indonesia</p>
            <p class="tech-stack">Powered by IndoBERT & Machine Learning</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Progress bar dengan animasi
        st.markdown('<div class="progress-container">', unsafe_allow_html=True)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simulasi loading
        loading_steps = [
            (10, "Initializing..."),
            (30, "Loading data..."),
            (50, "Preparing models..."),
            (70, "Configuring visualizations..."),
            (90, "Finalizing..."),
            (100, "Ready")
        ]
        
        for progress, status in loading_steps:
            progress_bar.progress(progress)
            status_text.text(status)
            time.sleep(0.3)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Tombol masuk
        st.markdown('<div class="button-container">', unsafe_allow_html=True)
        if st.button("ENTER DASHBOARD", use_container_width=True):
            st.session_state.show_intro = False
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.stop()  # Stop execution, jangan tampilkan dashboard

# KONFIGURASI HALAMAN
st.set_page_config(
    page_title="Dashboard Sentimen M-Banking",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)
# LOAD DATA

@st.cache_data
def load_data():
    df = pd.read_csv('data/processed/final_data_with_sentiment.csv')
    df['at'] = pd.to_datetime(df['at'], errors='coerce')
    return df

try:
    df = load_data()
except:
    st.error("❌ Error: File data tidak ditemukan.")
    st.stop()


# SIDEBAR - FUTURISTIK MINIMALIS
# LOAD PREDICTION MODEL (Untuk Live Prediction)
@st.cache_resource
def load_prediction_model():
    """Load model dengan auto-detect path"""
    try:
        import os
        import glob
        
        # Coba berbagai kemungkinan path
        possible_paths = [
            './models/indobert_banking_model',
            'models/indobert_banking_model',
            '../models/indobert_banking_model',
            os.path.join(os.path.dirname(__file__), '..', 'models', 'indobert_banking_model')
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                # Cek apakah ada file model
                if os.path.exists(os.path.join(path, 'config.json')):
                    model_path = path
                    print(f"✅ Found model at: {path}")
                    break
        
        if model_path is None:
            st.error("❌ Model tidak ditemukan di semua path yang dicoba")
            return None
        
        # Load tokenizer & model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        st.success(f"✅ Model loaded on {device}")
        
        return {'model': model, 'tokenizer': tokenizer, 'device': device}
    
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

# Load model (optional - user bisa enable/disable)
with st.sidebar:
    st.markdown("---")
    st.markdown("**🤖 LIVE PREDICTION**")
    use_live_prediction = st.checkbox("Enable Live Prediction", value=False)
    
    predictor = None
    if use_live_prediction:
        with st.spinner("Loading model..."):
            predictor = load_prediction_model()
            if predictor:
                st.success("✅ Model loaded!")
            else:
                st.error("❌ Model not found")


# SIDEBAR - PROFESSIONAL

with st.sidebar:
    st.markdown("""
    <div style='
        background: linear-gradient(135deg, #2C5282 0%, #4A5568 100%);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 25px;
        border: 1px solid rgba(44, 82, 130, 0.4);
    '>
        <h2 style='
            color: white;
            font-size: 1.4rem;
            font-weight: 600;
            margin: 0;
            letter-spacing: 0.5px;
        '>FILTER</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Filter Bank - FIXED LABEL
    selected_banks = st.multiselect(
        label="Pilih Bank",  # ← LABEL DITAMBAHKAN
        options=sorted(df['sumber_bank'].unique()),
        default=sorted(df['sumber_bank'].unique()),
        label_visibility="visible"  # ← VISIBILITY DIUBAH
    )
    
    st.markdown("---")
    
    # Filter Sentimen - FIXED LABEL
    selected_sentiment = st.multiselect(
        label="Pilih Sentimen",  # ← LABEL DITAMBAHKAN
        options=['positive', 'negative', 'neutral'],
        default=['positive', 'negative'],
        label_visibility="visible"  # ← VISIBILITY DIUBAH
    )
    
    st.markdown("---")
    
    # Filter Tanggal - FIXED LABEL
    st.markdown("**📅 Tanggal**")
    min_date = df['at'].dropna().min()
    max_date = df['at'].dropna().max()
    
    if pd.notna(min_date) and pd.notna(max_date):
        start_date, end_date = st.date_input(
            label="Pilih Tanggal",  # ← LABEL DITAMBAHKAN
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date(),
            label_visibility="visible"  # ← VISIBILITY DIUBAH
        )
    else:
        start_date, end_date = None, None
        st.warning("⚠️ Tidak ada data tanggal")
    
    st.markdown("---")
    
    # Reset Button
    if st.button("🔄 RESET FILTER", use_container_width=True, type="primary"):
        st.session_state.clear()
        st.rerun()
    
    # Info Box
    try:
        st.markdown("""
        <div style='
            background: rgba(44, 82, 130, 0.1);
            border-left: 3px solid #2C5282;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
        '>
            <p style='
                color: #E2E8F0;
                font-size: 0.85rem;
                margin: 0;
                line-height: 1.6;
            '>
                <strong>📊 Stats</strong><br>
                Total Data: <span style='color: #38A169;'>{:,}</span> reviews<br>
                Date Range: <span style='color: #3182CE;'>{}</span>
            </p>
        </div>
        """.format(len(df), f"{min_date.strftime('%d/%m')} - {max_date.strftime('%d/%m')}"), 
        unsafe_allow_html=True)
    except:
        st.info("📊 Stats tidak tersedia")

    


# APPLY FILTERS
if selected_banks and selected_sentiment and start_date and end_date:
    df_filtered = df[
        (df['sumber_bank'].isin(selected_banks)) &
        (df['sentiment_pred'].isin(selected_sentiment)) &
        (df['at'] >= pd.to_datetime(start_date)) &
        (df['at'] <= pd.to_datetime(end_date))
    ]
else:
    df_filtered = df.copy()




# HEADER - PROFESSIONAL
st.markdown("""
<div style='text-align: center; padding: 35px; background: linear-gradient(135deg, rgba(44, 82, 130, 0.15) 0%, rgba(74, 85, 104, 0.15) 100%); border-radius: 15px; margin-bottom: 30px; border: 1px solid rgba(44, 82, 130, 0.4);'>
    <h1 style='font-size: 3.2rem; font-weight: 700; color: #FFFFFF; margin: 0; letter-spacing: -0.5px;'>CUSTOMER SENTIMENT DASHBOARD</h1>
    <p style='font-size: 1.2rem; color: #94A3B8; margin: 15px 0 0 0; font-weight: 400;'>Analisis Sentimen Mobile Banking Indonesia</p>
    <p style='font-size: 0.85rem; color: #64748B; margin: 12px 0 0 0; font-weight: 400;'>INDOBERT • MACHINE LEARNING</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# KPI METRICS
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_reviews = len(df_filtered)
    st.metric(label="📝 Total Reviews", value=f"{total_reviews:,}")

with col2:
    avg_score = df_filtered['score'].mean()
    st.metric(label="⭐ Average Score", value=f"{avg_score:.2f}")

with col3:
    positive_pct = (df_filtered['sentiment_pred'] == 'positive').mean() * 100 if len(df_filtered) > 0 else 0
    st.metric(label="😊 Positive", value=f"{positive_pct:.1f}%")

with col4:
    negative_pct = (df_filtered['sentiment_pred'] == 'negative').mean() * 100 if len(df_filtered) > 0 else 0
    st.metric(label="😠 Negative", value=f"{negative_pct:.1f}%")

st.markdown("---")


# TABS
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([    "📊 Overview",
    "🏦 Per Bank",
    "🔥 Top Issues",
    "📈 Trend",
    "💬 Reviews",
    "🤖 Live Prediction"  # ← TAB BARU
])


# TAB 1: OVERVIEW
with tab1:
    st.markdown('<p class="sub-header">Sentiment Overview</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        sentiment_counts = df_filtered['sentiment_pred'].value_counts()
        if len(sentiment_counts) > 0:
            fig_pie = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                hole=0.4,
                color=sentiment_counts.index,
                color_discrete_map={
                    'positive': COLORS['accent1'],
                    'negative': COLORS['secondary'],
                    'neutral': COLORS['accent2']
                }
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label',
                textfont=dict(size=12, color='white'),
                marker=dict(line=dict(color='#0F0A1F', width=2)))
            fig_pie.update_layout(height=400, showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'))
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.warning("⚠️ Tidak ada data")
    
    with col2:
        bank_sentiment = df_filtered.groupby(['sumber_bank', 'sentiment_pred']).size().unstack(fill_value=0)
        if len(bank_sentiment) > 0:
            fig_bar = px.bar(
                bank_sentiment.reset_index().melt(id_vars='sumber_bank'),
                x='sumber_bank', y='value', color='sentiment_pred',
                barmode='stack',
                color_discrete_map={
                    'positive': COLORS['accent1'],
                    'negative': COLORS['secondary'],
                    'neutral': COLORS['accent2']
                })
            fig_bar.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
                xaxis=dict(title='Bank', tickfont=dict(color='white')),
                yaxis=dict(title='Jumlah', tickfont=dict(color='white')),
                legend=dict(font=dict(color='white')))
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("⚠️ Tidak ada data")


# TAB 2: PER BANK
with tab2:
    st.markdown('<p class="sub-header">Bank Performance</p>', unsafe_allow_html=True)
    
    bank_stats = df_filtered.groupby('sumber_bank').agg({
        'score': 'mean',
        'sentiment_pred': lambda x: (x == 'positive').mean() * 100,
        'userName': 'count'
    }).round(2)
    
    bank_stats.columns = ['Avg Score', '% Positive', 'Total Reviews']
    bank_stats = bank_stats.sort_values('% Positive', ascending=False)
    
    st.dataframe(bank_stats, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_score = px.bar(
            bank_stats.reset_index(), x='sumber_bank', y='Avg Score',
            color='Avg Score',
            color_continuous_scale=[(0, COLORS['secondary']), (0.5, COLORS['accent2']), (1, COLORS['accent1'])])
        fig_score.update_yaxes(range=[0, 5])
        fig_score.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
            xaxis=dict(tickfont=dict(color='white')),
            yaxis=dict(tickfont=dict(color='white')))
        st.plotly_chart(fig_score, use_container_width=True)
    
    with col2:
        fig_positive = px.bar(
            bank_stats.reset_index(), x='sumber_bank', y='% Positive',
            color='% Positive',
            color_continuous_scale=[(0, COLORS['secondary']), (0.5, COLORS['accent2']), (1, COLORS['accent1'])])
        fig_positive.update_yaxes(range=[0, 100])
        fig_positive.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
            xaxis=dict(tickfont=dict(color='white')),
            yaxis=dict(tickfont=dict(color='white')))
        st.plotly_chart(fig_positive, use_container_width=True)


# TAB 3: TOP COMPLAINTS

with tab3:
    st.markdown('<p class="sub-header">Top Complaints</p>', unsafe_allow_html=True)
    
    complaint_keywords = {
        'Login/Verifikasi': ['login', 'verifikasi', 'otp', 'sms', 'kode', 'password', 'pin'],
        'Transaksi Gagal': ['gagal', 'error', 'tidak berhasil', 'transaksi gagal'],
        'Saldo Terpotong': ['saldo', 'potong', 'terpotong', 'berkurang'],
        'QRIS': ['qris', 'qr code', 'scan qr'],
        'Aplikasi Lemot': ['lemot', 'lambat', 'loading', 'hang'],
        'Layar Putih': ['layar putih', 'blank', 'ngeblank']
    }
    
    def count_keywords(reviews, keywords):
        count = 0
        for review in reviews:
            if pd.notna(review) and any(kw in str(review).lower() for kw in keywords):
                count += 1
        return count
    
    negative_df = df_filtered[df_filtered['sentiment_pred'] == 'negative']
    topics_data = []
    
    for bank in selected_banks:
        bank_negative = negative_df[negative_df['sumber_bank'] == bank]['content']
        for topic, keywords in complaint_keywords.items():
            count = count_keywords(bank_negative, keywords)
            if count > 0:
                topics_data.append({'bank': bank, 'topic': topic, 'count': count})
    
    topics_df = pd.DataFrame(topics_data)
    
    if len(topics_df) > 0:
        fig_complaints = px.bar(
            topics_df, x='count', y='topic', color='bank',
            orientation='h', barmode='group',
            color_discrete_map={'BCA': COLORS['accent3'], 'BNI': COLORS['accent2'], 'BRI': COLORS['accent1']})
        fig_complaints.update_layout(height=500, paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
            xaxis=dict(tickfont=dict(color='white')),
            yaxis=dict(tickfont=dict(color='white')),
            legend=dict(font=dict(color='white')))
        st.plotly_chart(fig_complaints, use_container_width=True)
        
        st.markdown("### 📋 Detail")
        st.dataframe(topics_df, use_container_width=True)
    else:
        st.warning("⚠️ Tidak ada data complaints")


# TAB 4: TREND ANALYSIS

with tab4:
    st.markdown('<p class="sub-header">Sentiment Trend</p>', unsafe_allow_html=True)
    
    df_trend = df_filtered.dropna(subset=['at']).copy()
    df_trend['month'] = df_trend['at'].dt.to_period('M').astype(str)
    
    if len(df_trend) > 0:
        trend_data = df_trend.groupby(['sumber_bank', 'month']).agg({
            'sentiment_pred': lambda x: (x == 'positive').sum() / len(x) * 100
        }).reset_index()
        trend_data.columns = ['sumber_bank', 'month', 'positive_pct']
        
        if len(trend_data) > 0:
            fig_trend = px.line(
                trend_data, x='month', y='positive_pct', color='sumber_bank',
                markers=True, line_shape='spline',
                color_discrete_map={'BCA': COLORS['accent3'], 'BNI': COLORS['accent2'], 'BRI': COLORS['accent1']})
            fig_trend.update_layout(height=450, paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
                xaxis=dict(tickfont=dict(color='white'), tickangle=45),
                yaxis=dict(tickfont=dict(color='white')),
                legend=dict(font=dict(color='white')))
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.warning("⚠️ Tidak ada data trend")
    else:
        st.warning("⚠️ Tidak ada data tanggal")


# TAB 5: SAMPLE REVIEWS

with tab5:
    st.markdown('<p class="sub-header">Recent Reviews</p>', unsafe_allow_html=True)
    
    sentiment_filter = st.selectbox("Filter", options=['All', 'positive', 'negative'])
    
    if sentiment_filter != 'All':
        sample_df = df_filtered[df_filtered['sentiment_pred'] == sentiment_filter]
    else:
        sample_df = df_filtered
    
    st.write(f"Menampilkan {min(10, len(sample_df))} dari {len(sample_df)} reviews")
    
    for idx, row in sample_df.head(10).iterrows():
        with st.expander(f"{row['sentiment_pred'].upper()} - {row['sumber_bank']} - Score: {row['score']}⭐"):
            st.write(f"**Review:** {row['content']}")
            st.write(f"**Tanggal:** {row['at']}")
            if 'confidence_score' in row:
                st.write(f"**Confidence:** {row['confidence_score']:.2f}")


# TAB 6: LIVE PREDICTION (WITH TOPIC DETECTION)
with tab6:
    st.markdown('<p class="sub-header">🤖 Live Sentiment & Topic Prediction</p>', unsafe_allow_html=True)
    
    # Define complaint keywords (sama seperti di 04_business_insight.py)
    complaint_keywords = {
        'QRIS': ['qris', 'qr code', 'scan qr', 'pake qr'],
        'Login/Verifikasi': ['login', 'verifikasi', 'otp', 'sms', 'kode', 'password', 'pin', 'masuk'],
        'Transaksi Gagal': ['gagal', 'error', 'tidak berhasil', 'transaksi gagal', 'eror'],
        'Saldo Terpotong': ['saldo', 'potong', 'terpotong', 'berkurang', 'keluar'],
        'Aplikasi Lemot': ['lemot', 'lambat', 'loading', 'hang', 'freeze', 'slow'],
        'Layar Putih/Blank': ['layar putih', 'blank', 'ngeblank', 'ngebleng', 'white screen'],
        'Notifikasi': ['notifikasi', 'notif', 'pesan', 'sms notifikasi'],
        'Mutasi/Riwayat': ['mutasi', 'riwayat', 'bukti transaksi', 'histori'],
        'Keamanan/Data': ['data', 'privasi', 'aman', 'keamanan', 'bocor', 'blokir'],
        'Customer Service': ['cs', 'customer service', 'hallo bca', 'call center', 'admin'],
        'Biaya Admin': ['biaya', 'admin', 'potongan', 'tarif', 'cost'],
        'Update Aplikasi': ['update', 'instal', 'uninstall', 'download', 'install ulang']
    }
    
    def detect_topics(text):
        """Detect topik/masalah dari teks"""
        text_lower = text.lower()
        detected_topics = []
        
        for topic, keywords in complaint_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    detected_topics.append(topic)
                    break  # Satu topik cukup sekali
        
        return detected_topics
    
    # Info box
    st.info("""
    ### 📝 Cara Menggunakan
    1. **Masukkan review/komen** tentang mobile banking
    2. Klik **🔮 PREDIKSI**
    3. Lihat hasil: **Sentimen + Masalah Utama**
    
    **Contoh:** "Waktu QRIS gagal, saldo terpotong!" → Deteksi: QRIS, Saldo Terpotong
    """)
    
    st.markdown("---")
    
    # Input form
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_input = st.text_area(
            "💬 Masukkan Review/Komen:",
            placeholder="Contoh: Waktu QRIS gagal, saldo terpotong tapi transaksi tidak berhasil...",
            height=100,
            key="live_input"
        )
    
    with col2:
        predict_button = st.button("🔮 PREDIKSI", use_container_width=True, type="primary", key="live_predict")
    
    # Prediction result
    if predict_button and user_input:
        with st.spinner("🤖 Menganalisis..."):
            # 1. Detect Topics
            detected_topics = detect_topics(user_input)
            
            # 2. Predict Sentiment (dengan model jika ada, atau keyword-based)
            if predictor:
                # Pakai model IndoBERT
                def clean_text(text):
                    text = text.lower()
                    text = re.sub(r'http\S+|www\S+', '', text)
                    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
                    text = ' '.join(text.split())
                    return text
                
                cleaned_text = clean_text(user_input)
                
                inputs = predictor['tokenizer'](
                    cleaned_text,
                    padding='max_length',
                    truncation=True,
                    max_length=128,
                    return_tensors='pt'
                ).to(predictor['device'])
                
                with torch.no_grad():
                    outputs = predictor['model'](**inputs)
                    probabilities = torch.softmax(outputs.logits, dim=1)
                    prediction = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][prediction].item()
                
                label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
                sentiment = label_mapping[prediction]
                
                model_type = "IndoBERT (AI)"
            else:
                # Keyword-based fallback
                positive_words = ['mudah', 'cepat', 'puas', 'bagus', 'baik', 'suka', 'mantap', 'keren', 'helpful', 'recommended', 'sukses', 'berhasil']
                negative_words = ['gagal', 'lemot', 'error', 'kecewa', 'buruk', 'susah', 'jelek', 'marah', 'masalah', 'kecewa']
                
                text_lower = user_input.lower()
                pos_count = sum(1 for word in positive_words if word in text_lower)
                neg_count = sum(1 for word in negative_words if word in text_lower)
                
                if pos_count > neg_count:
                    sentiment = 'positive'
                    confidence = min(0.6 + (pos_count * 0.1), 0.95)
                elif neg_count > pos_count:
                    sentiment = 'negative'
                    confidence = min(0.6 + (neg_count * 0.1), 0.95)
                else:
                    sentiment = 'neutral'
                    confidence = 0.5
                
                model_type = "Keyword-based (Demo)"
            
            # Display Results
            st.markdown("---")
            
            # Row 1: Sentiment & Topics
            st.markdown("### 📊 Hasil Analisis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                emoji = "🟢" if sentiment == 'positive' else "🔴" if sentiment == 'negative' else "🟡"
                st.metric(
                    label="Sentimen",
                    value=f"{emoji} {sentiment.upper()}"
                )
            
            with col2:
                st.metric(
                    label="Confidence",
                    value=f"{confidence*100:.1f}%"
                )
            
            with col3:
                st.metric(
                    label="Model",
                    value=model_type
                )
            
            # Row 2: Detected Topics (MASALAH UTAMA)
            st.markdown("---")
            st.markdown("### 🎯 Masalah Utama Terdeteksi")
            
            if len(detected_topics) > 0:
                # Display topics dengan icon
                topic_icons = {
                    'QRIS': '📱',
                    'Login/Verifikasi': '🔐',
                    'Transaksi Gagal': '❌',
                    'Saldo Terpotong': '💰',
                    'Aplikasi Lemot': '🐌',
                    'Layar Putih/Blank': '⬜',
                    'Notifikasi': '🔔',
                    'Mutasi/Riwayat': '📋',
                    'Keamanan/Data': '🔒',
                    'Customer Service': '🎧',
                    'Biaya Admin': '💵',
                    'Update Aplikasi': '🔄'
                }
                
                # Display sebagai badges
                topic_badges = ""
                for topic in detected_topics:
                    icon = topic_icons.get(topic, '📌')
                    topic_badges += f"`{icon} {topic}` "
                
                st.success(f"**Topik Terdeteksi:** {topic_badges}", icon="🎯")
                
                # Priority recommendation
                st.markdown("### 💡 Rekomendasi")
                
                if sentiment == 'negative' and len(detected_topics) > 0:
                    if 'Saldo Terpotong' in detected_topics or 'Keamanan/Data' in detected_topics:
                        st.error(f"""
                        🚨 **HIGH PRIORITY**
                        
                        **Masalah:** {', '.join(detected_topics)}
                        
                        **Action:** 
                        - ⚠️ Ini masalah keuangan/keamanan - perlu tindak lanjut SEGERA
                        - 📞 Hubungi customer service
                        - 🔍 Investigasi transaksi
                        """)
                    elif 'Transaksi Gagal' in detected_topics or 'QRIS' in detected_topics:
                        st.warning(f"""
                        ⚠️ **MEDIUM PRIORITY**
                        
                        **Masalah:** {', '.join(detected_topics)}
                        
                        **Action:**
                        - 🔧 Cek sistem transaksi
                        - 📝 Log error untuk investigasi
                        - 💬 Follow up dengan user
                        """)
                    else:
                        st.info(f"""
                        📌 **LOW PRIORITY**
                        
                        **Masalah:** {', '.join(detected_topics)}
                        
                        **Action:**
                        - 📊 Monitor trend complaints
                        - 🔄 Improve di update berikutnya
                        """)
                elif sentiment == 'positive':
                    st.success(f"""
                    ✅ **REVIEW POSITIF**
                    
                    **Topik:** {', '.join(detected_topics) if detected_topics else 'General'}
                    
                    **Action:**
                    - ⭐ Bisa jadi testimonial
                    - 📣 Share ke marketing
                    - 👍 Thank the user
                    """)
                else:
                    st.info("📝 Review neutral - monitor untuk trend")
            
            else:
                st.warning("""
                ⚠️ **Tidak ada topik spesifik terdeteksi**
                
                Review ini bersifat general/tidak menyebutkan masalah spesifik.
                
                **Suggestion:** Minta user memberikan detail lebih spesifik.
                """)
            
            # Row 3: Probability Distribution (jika pakai model)
            if predictor:
                st.markdown("---")
                st.markdown("### 📈 Probability Distribution")
                
                prob_df = pd.DataFrame({
                    'Sentimen': ['Negative', 'Neutral', 'Positive'],
                    'Probability': [
                        probabilities[0][0].item() * 100,
                        probabilities[0][1].item() * 100,
                        probabilities[0][2].item() * 100
                    ]
                })
                
                fig_prob = px.bar(
                    prob_df,
                    x='Sentimen',
                    y='Probability',
                    color='Probability',
                    color_continuous_scale='RdYlGn',
                    text_auto='.1f'
                )
                fig_prob.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig_prob, use_container_width=True)
            
# FOOTER

st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: {COLORS["secondary"]}; font-size: 0.9rem;'>
    <p style='opacity: 0.7;'>Dashboard Sentimen M-Banking</p>
</div>
""", unsafe_allow_html=True)