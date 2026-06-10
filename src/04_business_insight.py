import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import matplotlib
import os

# Set style dan font
plt.style.use('seaborn-v0_8-whitegrid')
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# Create directory
os.makedirs('reports', exist_ok=True)


# LOAD DATA

print("="*80)
print("LOADING DATA FOR VISUALIZATION")
print("="*80)

df = pd.read_csv('data/processed/final_data_with_sentiment.csv')
print(f"Loaded {len(df)} reviews")

# Convert date
df['at'] = pd.to_datetime(df['at'], errors='coerce')
df['month'] = df['at'].dt.to_period('M').astype(str)


# DATA COMPLAINTS - PRIORITY BERDASARKAN JUMLAH
print("\nCALCULATING PRIORITY BASED ON COMPLAINT COUNTS...")

complaints_data_raw = {
    'Bank': ['BCA', 'BCA', 'BCA', 'BCA', 'BNI', 'BNI', 'BRI', 'BRI'],
    'Topik': ['Login/Verifikasi', 'Saldo Terpotong', 'Transaksi Gagal', 'QRIS', 
              'Login/Verifikasi', 'Transaksi Gagal', 
              'Login/Verifikasi', 'Layar Putih/Blank'],
    'Jumlah': [1020, 1090, 980, 580, 440, 460, 750, 320]
}

df_complaints = pd.DataFrame(complaints_data_raw)

# Tentukan prioritas berdasarkan jumlah complaints
def tentukan_prioritas(jumlah):
    if jumlah >= 900:
        return 'Tinggi'
    elif jumlah >= 400:
        return 'Sedang'
    else:
        return 'Rendah'

df_complaints['Prioritas'] = df_complaints['Jumlah'].apply(tentukan_prioritas)

print("\nPRIORITY CLASSIFICATION:")
print(df_complaints[['Bank', 'Topik', 'Jumlah', 'Prioritas']].to_string(index=False))

# Warna Priority (WAJIB)
priority_colors = {
    'Tinggi': '#e74c3c',    # Merah
    'Sedang': '#f39c12',    # Kuning/Orange
    'Rendah': '#2ecc71'     # Hijau
}

# Warna Bank (Bebas)
bank_colors = {
    'BCA': '#3498db',    # Biru
    'BNI': '#f39c12',    # Orange/Kuning
    'BRI': '#2ecc71'     # Hijau
}


# VISUALISASI 1: TOP COMPLAINTS PER BANK (3 PANEL)
print("\nGenerating Visualization 1: Top Complaints Per Bank...")

fig, axes = plt.subplots(1, 3, figsize=(16, 7))
fig.suptitle('TOP COMPLAINTS PER BANK - MOBILE BANKING', 
             fontsize=15, fontweight='bold', y=0.98)

# BCA
bca_data = df_complaints[df_complaints['Bank'] == 'BCA'].sort_values('Jumlah', ascending=True)
colors_bca = [priority_colors[p] for p in bca_data['Prioritas']]
axes[0].barh(bca_data['Topik'], bca_data['Jumlah'], color=colors_bca, alpha=0.8, edgecolor='black')
axes[0].set_xlabel('Jumlah Complaints', fontsize=10)
axes[0].set_title('BCA - PRIORITAS TINGGI', fontsize=12, fontweight='bold', color='#e74c3c')
axes[0].invert_yaxis()
axes[0].grid(axis='x', alpha=0.3)
axes[0].set_xlim(0, max(bca_data['Jumlah']) * 1.2)
for i, v in enumerate(bca_data['Jumlah']):
    axes[0].text(v + 30, i, f'{v:,}', color='black', fontweight='bold', va='center', fontsize=9)

# BNI
bni_data = df_complaints[df_complaints['Bank'] == 'BNI'].sort_values('Jumlah', ascending=True)
colors_bni = [priority_colors[p] for p in bni_data['Prioritas']]
axes[1].barh(bni_data['Topik'], bni_data['Jumlah'], color=colors_bni, alpha=0.8, edgecolor='black')
axes[1].set_xlabel('Jumlah Complaints', fontsize=10)
axes[1].set_title('BNI - PRIORITAS SEDANG', fontsize=12, fontweight='bold', color='#f39c12')
axes[1].invert_yaxis()
axes[1].grid(axis='x', alpha=0.3)
axes[1].set_xlim(0, max(bni_data['Jumlah']) * 1.2)
for i, v in enumerate(bni_data['Jumlah']):
    axes[1].text(v + 30, i, f'{v:,}', color='black', fontweight='bold', va='center', fontsize=9)

# BRI
bri_data = df_complaints[df_complaints['Bank'] == 'BRI'].sort_values('Jumlah', ascending=True)
colors_bri = [priority_colors[p] for p in bri_data['Prioritas']]
axes[2].barh(bri_data['Topik'], bri_data['Jumlah'], color=colors_bri, alpha=0.8, edgecolor='black')
axes[2].set_xlabel('Jumlah Complaints', fontsize=10)
axes[2].set_title('BRI - PRIORITAS RENDAH', fontsize=12, fontweight='bold', color='#2ecc71')
axes[2].invert_yaxis()
axes[2].grid(axis='x', alpha=0.3)
axes[2].set_xlim(0, max(bri_data['Jumlah']) * 1.2)
for i, v in enumerate(bri_data['Jumlah']):
    axes[2].text(v + 30, i, f'{v:,}', color='black', fontweight='bold', va='center', fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('reports/01_top_complaints_per_bank.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: reports/01_top_complaints_per_bank.png")


# VISUALISASI 2: TOTAL COMPLAINTS PER BANK
print("\nGenerating Visualization 2: Total Complaints Per Bank...")

fig, ax = plt.subplots(figsize=(10, 6))

banks = ['BCA', 'BNI', 'BRI']
total_complaints = [df_complaints[df_complaints['Bank']==bank]['Jumlah'].sum() for bank in banks]

# Warna berdasarkan priority total
priority_total = []
for bank in banks:
    bank_data = df_complaints[df_complaints['Bank']==bank]
    avg_priority = bank_data['Jumlah'].mean()
    if avg_priority >= 900:
        priority_total.append('Tinggi')
    elif avg_priority >= 400:
        priority_total.append('Sedang')
    else:
        priority_total.append('Rendah')

colors = [priority_colors[p] for p in priority_total]

bars = ax.bar(banks, total_complaints, color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)

ax.set_title('TOTAL COMPLAINTS PER BANK', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Bank', fontsize=11)
ax.set_ylabel('Total Complaints', fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, max(total_complaints) * 1.15)

# Add value labels di atas bar
for bar, total in zip(bars, total_complaints):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 50,
            f'{total:,}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add priority label di tengah bar
for i, (bank, label) in enumerate(zip(banks, priority_total)):
    ax.text(i, total_complaints[i]/2, f'PRIO\n{label}', 
            ha='center', va='center', fontsize=10, fontweight='bold', 
            color='white', alpha=0.95)

plt.tight_layout()
plt.savefig('reports/02_total_complaints_priority.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: reports/02_total_complaints_priority.png")


# VISUALISASI 3: POSITIVE SENTIMENT TREND OVER TIME
print("\nGenerating Visualization 3: Positive Sentiment Trend...")

common_period_start = datetime(2026, 1, 1)
common_period_end = datetime(2026, 2, 28)
df_common = df[(df['at'] >= common_period_start) & (df['at'] <= common_period_end)].copy()
df_valid = df_common.dropna(subset=['at'])

if len(df_valid) > 0:
    trend_data = df_valid.groupby(['sumber_bank', 'month', 'sentiment_pred']).size().unstack(fill_value=0)
    
    if 'positive' in trend_data.columns:
        trend_data_pct = trend_data['positive'] / trend_data.sum(axis=1) * 100
        
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_data = trend_data_pct.unstack(0)
        
        # Plot dengan warna bank
        for bank in plot_data.columns:
            ax.plot(plot_data.index, plot_data[bank], marker='o', linewidth=2, 
                   markersize=7, label=bank, color=bank_colors[bank])
        
        ax.set_title('Positive Sentiment Trend Over Time (Jan-Feb 2026)', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Month', fontsize=11)
        ax.set_ylabel('Percentage Positive (%)', fontsize=11)
        ax.legend(title='Bank', loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('reports/03_sentiment_trend.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Saved: reports/03_sentiment_trend.png")


# VISUALISASI 4: SENTIMENT DISTRIBUTION PER BANK (%)
print("\nGenerating Visualization 4: Sentiment Distribution...")

df_2class = df[df['sentiment_pred'].isin(['positive', 'negative'])].copy()

# PERBAIKAN 1: Figsize lebih besar
fig, ax = plt.subplots(figsize=(12, 7))

sentiment_by_bank = df_2class.groupby(['sumber_bank', 'sentiment_pred']).size().unstack(fill_value=0)
sentiment_by_bank_pct = sentiment_by_bank.div(sentiment_by_bank.sum(axis=1), axis=0) * 100

# Urutkan bank
bank_order = ['BCA', 'BNI', 'BRI']
sentiment_by_bank_pct = sentiment_by_bank_pct.reindex(bank_order)

# Warna konsisten
colors = ['#e74c3c', '#2ecc71']  # Red for Negative, Green for Positive

# Plot
sentiment_by_bank_pct.plot(
    kind='bar', 
    stacked=True, 
    ax=ax, 
    color=colors,
    width=0.5,  # PERBAIKAN 2: Bar lebih kecil agar ada spacing
    alpha=0.9, 
    edgecolor='black',
    linewidth=1.5
)

ax.set_title('Sentiment Distribution per Bank (%)', fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Bank', fontsize=12)
ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_xticklabels(sentiment_by_bank_pct.index, rotation=0, fontsize=11)

# PERBAIKAN 3: Legend di luar plot (bawah)
ax.legend(['Negative', 'Positive'], loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)

ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3)

# PERBAIKAN 4: Label posisi lebih baik dengan offset
for i, bank in enumerate(sentiment_by_bank_pct.index):
    neg_pct = sentiment_by_bank_pct.loc[bank, 'negative']
    pos_pct = sentiment_by_bank_pct.loc[bank, 'positive']
    
    # Label negative - hanya jika > 20% agar tidak terlalu kecil
    if neg_pct > 20:
        ax.text(i, neg_pct/2, f'{neg_pct:.0f}%', ha='center', va='center', 
                fontsize=12, fontweight='bold', color='white')
    
    # Label positive - hanya jika > 20%
    if pos_pct > 20:
        ax.text(i, neg_pct + pos_pct/2, f'{pos_pct:.0f}%', ha='center', va='center', 
                fontsize=12, fontweight='bold', color='white')
    
    # Jika persentase kecil, taruh label di luar bar
    if neg_pct <= 20 and neg_pct > 0:
        ax.text(i, 5, f'{neg_pct:.0f}%', ha='center', va='bottom', 
                fontsize=10, fontweight='bold', color='black')
    if pos_pct <= 20 and pos_pct > 0:
        ax.text(i, neg_pct + 5, f'{pos_pct:.0f}%', ha='center', va='bottom', 
                fontsize=10, fontweight='bold', color='black')

# PERBAIKAN 5: Tight layout dengan rect agar legend tidak terpotong
plt.tight_layout(rect=[0, 0, 0.95, 1])
plt.savefig('reports/04_sentiment_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: reports/04_sentiment_distribution.png")


# VISUALISASI 5: TOP COMPLAINT TOPICS PER BANK
print("\nGenerating Visualization 5: Top Complaint Topics...")

complaint_keywords = {
    'Login/Verifikasi': ['login', 'verifikasi', 'otp', 'sms', 'kode', 'password', 'pin'],
    'Transaksi Gagal': ['gagal', 'error', 'tidak berhasil', 'transaksi gagal'],
    'Saldo Terpotong': ['saldo', 'potong', 'terpotong', 'berkurang'],
    'QRIS': ['qris', 'qr code', 'scan qr'],
    'Notifikasi': ['notifikasi', 'notif', 'pesan'],
    'Mutasi/Riwayat': ['mutasi', 'riwayat', 'bukti transaksi'],
    'Layar Putih/Blank': ['layar putih', 'blank', 'ngeblank'],
    'Keamanan/Data': ['data', 'privasi', 'aman', 'keamanan'],
    'Customer Service': ['cs', 'customer service', 'call center'],
    'Biaya Admin': ['biaya', 'admin', 'potongan'],
    'Aplikasi Lemot': ['lemot', 'lambat', 'loading', 'hang'],
    'Update Aplikasi': ['update', 'instal', 'uninstall']
}

def count_keywords(reviews, keywords):
    count = 0
    for review in reviews:
        if pd.notna(review) and any(kw in str(review).lower() for kw in keywords):
            count += 1
    return count

negative_df = df[df['sentiment_pred'] == 'negative']
topics_data = []

for bank in df['sumber_bank'].unique():
    bank_negative = negative_df[negative_df['sumber_bank'] == bank]['content']
    for topic, keywords in complaint_keywords.items():
        count = count_keywords(bank_negative, keywords)
        topics_data.append({'bank': bank, 'topic': topic, 'count': count})

topics_df = pd.DataFrame(topics_data)

# Filter hanya topik dengan complaints > 0
topics_df_filtered = topics_df[topics_df['count'] > 0]

if len(topics_df_filtered) > 0:
    fig, ax = plt.subplots(figsize=(12, 8))
    pivot_data = topics_df_filtered.pivot(index='topic', columns='bank', values='count').fillna(0)
    
    # Plot dengan warna bank
    pivot_data.plot(kind='barh', ax=ax, 
                   color=[bank_colors[bank] for bank in pivot_data.columns],
                   alpha=0.8, edgecolor='black')
    
    ax.set_title('Top Complaint Topics per Bank (Negative Reviews)', 
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Number of Complaints', fontsize=11)
    ax.set_ylabel('Topic Category', fontsize=11)
    ax.legend(title='Bank', labels=list(pivot_data.columns), fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('reports/05_complaint_topics.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved: reports/05_complaint_topics.png")


# SUMMARY

print("\n" + "="*80)
print("ALL VISUALIZATIONS COMPLETE!")
print("="*80)
print("\nGenerated Files:")
print("   1. reports/01_top_complaints_per_bank.png")
print("   2. reports/02_total_complaints_priority.png")
print("   3. reports/03_sentiment_trend.png")
print("   4. reports/04_sentiment_distribution.png")
print("   5. reports/05_complaint_topics.png")
print("\nPriority Colors:")
print("   High   = Red (#e74c3c)")
print("   Medium = Yellow/Orange (#f39c12)")
print("   Low    = Green (#2ecc71)")
print("\nReady for presentation!")