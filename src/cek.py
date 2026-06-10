import pandas as pd

df = pd.read_csv('data/processed/final_data_with_sentiment.csv')

print(" SENTIMENT DISTRIBUTION CHECK")
print("="*60)
print(f"Total reviews: {len(df)}")
print(f"\nUnique values in sentiment_pred:")
print(df['sentiment_pred'].unique())
print(f"\nValue counts:")
print(df['sentiment_pred'].value_counts())

# Check per bank
print("\n📊 PER BANK CHECK:")
for bank in ['BCA', 'BNI', 'BRI']:
    bank_df = df[df['sumber_bank'] == bank]
    print(f"\n{bank}:")
    print(f"   Total: {len(bank_df)}")
    print(f"   Positive: {(bank_df['sentiment_pred'] == 'positive').sum()}")
    print(f"   Negative: {(bank_df['sentiment_pred'] == 'negative').sum()}")
    print(f"   Neutral: {(bank_df['sentiment_pred'] == 'neutral').sum()}")
    
    # Calculate percentage
    if len(bank_df) > 0:
        pos_pct = (bank_df['sentiment_pred'] == 'positive').sum() / len(bank_df) * 100
        neg_pct = (bank_df['sentiment_pred'] == 'negative').sum() / len(bank_df) * 100
        print(f"   % Positive: {pos_pct:.1f}%")
        print(f"   % Negative: {neg_pct:.1f}%")