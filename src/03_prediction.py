import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import warnings
import os
warnings.filterwarnings('ignore')

os.makedirs('data/processed', exist_ok=True)
os.makedirs('reports', exist_ok=True)


# 1. LOAD MODEL
print("="*80)
print("🔮 STEP 3: SENTIMENT PREDICTION")
print("="*80)

MODEL_PATH = './models/indobert_banking_model'

print(f"\n📦 Loading model from: {MODEL_PATH}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

device = 0 if torch.cuda.is_available() else -1
print(f"🖥️  Device: {'GPU' if device == 0 else 'CPU'}")


# 2. CREATE PREDICTION PIPELINE
classifier = pipeline(
    'sentiment-analysis',
    model=model,
    tokenizer=tokenizer,
    device=device
)

print("✅ Model loaded successfully!")


# 3. LOAD FULL DATASET
print("\n" + "="*80)
print("📂 LOADING FULL DATASET")
print("="*80)

df = pd.read_csv('data/processed/prepared_data.csv')
print(f"📈 Total reviews to predict: {len(df)}")


# 4. DATA VALIDATION - FIX NON-STRING VALUES

print("\n" + "="*80)
print("🔍 VALIDATING DATA")
print("="*80)

# Check for non-string values
print(f"\n📊 Data types in 'teks_bersih':")
print(df['teks_bersih'].apply(type).value_counts())

# Fill NaN with empty string
df['teks_bersih'] = df['teks_bersih'].fillna('')

# Convert all to string
df['teks_bersih'] = df['teks_bersih'].astype(str)

# Check for empty strings
empty_count = (df['teks_bersih'] == '').sum()
print(f"\n⚠️  Empty reviews: {empty_count}")

# Check for very short reviews
short_count = (df['teks_bersih'].str.len() < 5).sum()
print(f"⚠️  Very short reviews (<5 chars): {short_count}")

print("✅ Data validation complete!")


# 5. BATCH PREDICTION
print("\n" + "="*80)
print("🔄 RUNNING PREDICTIONS")
print("="*80)

def predict_batch(texts, batch_size=32):
    """
    Predict sentiment in batches for efficiency
    """
    sentiments = []
    confidences = []
    
    for i in range(0, len(texts), batch_size):
        # Convert all texts to string and handle NaN/None
        batch = [str(text) if pd.notna(text) else "" for text in texts[i:i+batch_size]]
        results = classifier(batch)
        
        sentiments.extend([r['label'] for r in results])
        confidences.extend([r['score'] for r in results])
        
        if (i // batch_size) % 10 == 0:
            print(f"   Progress: {i}/{len(texts)} ({i/len(texts)*100:.1f}%)")
    
    return sentiments, confidences

# Run prediction
print("\n🚀 Starting batch prediction...")
df['sentiment_pred'], df['confidence_score'] = predict_batch(df['teks_bersih'])

print("✅ Prediction complete!")


# 6. MAP LABELS TO READABLE FORMAT
print("\n" + "="*80)
print("🏷️  MAPPING LABELS")
print("="*80)

# IndoBERT outputs: LABEL_0, LABEL_1, LABEL_2
label_mapping = {
    'LABEL_0': 'negative',
    'LABEL_1': 'neutral',
    'LABEL_2': 'positive'
}

df['sentiment_pred'] = df['sentiment_pred'].map(label_mapping)

print("\n📊 Prediction Distribution:")
print(df['sentiment_pred'].value_counts())


# 7. ADD BUSINESS COLUMNS
print("\n" + "="*80)
print("💼 ADDING BUSINESS METRICS")
print("="*80)

# Confidence level categorization
def confidence_level(score):
    if score >= 0.9:
        return 'Very High'
    elif score >= 0.75:
        return 'High'
    elif score >= 0.6:
        return 'Medium'
    else:
        return 'Low'

df['confidence_level'] = df['confidence_score'].apply(confidence_level)

# Risk flag for business (negative + high confidence = high priority)
df['priority_flag'] = df.apply(
    lambda x: 'HIGH' if (x['sentiment_pred'] == 'negative' and x['confidence_score'] >= 0.8) 
              else ('MEDIUM' if x['sentiment_pred'] == 'negative' else 'LOW'),
    axis=1
)

print("\n📊 Priority Flag Distribution:")
print(df['priority_flag'].value_counts())


# 8. SAVE FINAL DATASET
print("\n" + "="*80)
print("💾 SAVING FINAL DATASET")
print("="*80)

# Select business-relevant columns
output_columns = [
    'userName', 'score', 'at', 'content', 'teks_bersih', 
    'sumber_bank', 'label', 'sentiment_pred', 'confidence_score', 
    'confidence_level', 'priority_flag'
]

df_output = df[output_columns]
df_output.to_csv('data/processed/final_data_with_sentiment.csv', index=False)

print(f"✅ Saved: data/processed/final_data_with_sentiment.csv")
print(f"📈 Shape: {df_output.shape}")


# 9. SAMPLE PREDICTIONS
print("\n" + "="*80)
print("📝 SAMPLE PREDICTIONS")
print("="*80)

sample = df_output.sample(5, random_state=42)
for idx, row in sample.iterrows():
    print(f"\n📍 Review: {str(row['content'])[:100]}...")
    print(f"   Bank: {row['sumber_bank']}")
    print(f"   Score: {row['score']} ⭐")
    print(f"   Predicted: {row['sentiment_pred']} ({row['confidence_score']:.2f})")
    print(f"   Priority: {row['priority_flag']}")

print("\n" + "="*80)
print("✅ STEP 3 COMPLETE - Predictions Ready for Dashboard!")
print("="*80)