import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import os

# Create directories if not exist
os.makedirs('data/processed', exist_ok=True)
os.makedirs('reports', exist_ok=True)


# 1. LOAD DATA
print("="*80)
print("📊 STEP 1: DATA PREPARATION")
print("="*80)

# Load CSV dengan path yang benar
df = pd.read_csv('C:\\Users\\thori\\Documents\\Semester 4\\analisis bisnis\\m2\\project_sentimen_banking\\data\\raw\\dataset_lumayan_resik_iki.csv')

print(f"\n✅ Data loaded successfully!")
print(f"📈 Shape: {df.shape}")
print(f"\n📋 Columns: {df.columns.tolist()}")


# 2. DATA EXPLORATION
print("\n" + "="*80)
print("🔍 DATA EXPLORATION")
print("="*80)

# Cek distribusi score
print("\n⭐ Score Distribution:")
print(df['score'].value_counts().sort_index())

# Cek distribusi bank
print("\n🏦 Bank Distribution:")
print(df['sumber_bank'].value_counts())

# Cek sample data
print("\n📝 Sample Data:")
print(df[['teks_bersih', 'score', 'sumber_bank']].head(10))

# Cek missing values
print("\n❓ Missing Values:")
print(df.isnull().sum())


# 3. DATA CLEANING
print("\n" + "="*80)
print("🧹 DATA CLEANING")
print("="*80)

# Remove rows with missing score
df = df.dropna(subset=['score'])

# Convert score to numeric
df['score'] = pd.to_numeric(df['score'], errors='coerce')
df = df.dropna(subset=['score'])

# Convert score to integer
df['score'] = df['score'].astype(int)

print(f"✅ Data cleaned. Shape: {df.shape}")


# 4. CREATE SENTIMENT LABELS FROM SCORE
print("\n" + "="*80)
print("🏷️  CREATING SENTIMENT LABELS FROM SCORE")
print("="*80)

def label_from_score(score):
    """
    Mapping score bintang ke sentimen
    1-2 = Negative, 3 = Neutral, 4-5 = Positive
    """
    if score >= 4:
        return 'positive'
    elif score == 3:
        return 'neutral'
    else:
        return 'negative'

df['label'] = df['score'].apply(label_from_score)
df['label_code'] = df['label'].map({'negative': 0, 'neutral': 1, 'positive': 2})

print("\n📊 Sentiment Distribution:")
print(df['label'].value_counts())

# Visualisasi distribusi sentimen
plt.figure(figsize=(10, 6))
colors = ['#e74c3c', '#f39c12', '#2ecc71']
plt.bar(df['label'].value_counts().index, 
        df['label'].value_counts().values,
        color=colors)
plt.title('📊 Distribusi Sentimen dari Score', fontsize=14, fontweight='bold')
plt.xlabel('Sentimen')
plt.ylabel('Jumlah Review')
plt.savefig('reports/sentiment_distribution.png', dpi=300, bbox_inches='tight')
plt.show()


# 5. HANDLE CLASS IMBALANCE
print("\n" + "="*80)
print("⚖️  CHECKING CLASS IMBALANCE")
print("="*80)

label_counts = df['label'].value_counts()
imbalance_ratio = label_counts.max() / label_counts.min()
print(f"\n📈 Imbalance Ratio: {imbalance_ratio:.2f}x")

if imbalance_ratio > 3:
    print("⚠️  WARNING: Class imbalance detected! Will use stratified split.")
else:
    print("✅ Class distribution is balanced.")


# 6. SPLIT DATA FOR TRAINING
print("\n" + "="*80)
print("✂️  SPLITTING DATA FOR TRAINING")
print("="*80)

# Stratified split untuk maintain distribusi kelas
train_df, test_df = train_test_split(
    df, 
    test_size=0.2, 
    stratify=df['label_code'], 
    random_state=42
)

print(f"\n📚 Training Set: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
print(f"🧪 Testing Set: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")

# Save split data
train_df.to_csv('data/processed/train_data.csv', index=False)
test_df.to_csv('data/processed/test_data.csv', index=False)

print("\n✅ Data preparation complete!")
print("📁 Saved: train_data.csv, test_data.csv")


# 7. SAVE PREPARED DATA FOR MODELING
df.to_csv('data/processed/prepared_data.csv', index=False)
print("📁 Saved: prepared_data.csv")

print("\n" + "="*80)
print("✅ STEP 1 COMPLETE - Ready for Model Training!")
print("="*80)