import pandas as pd
import time
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer
)
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

# Create directories
os.makedirs('models/indobert_banking_model', exist_ok=True)
os.makedirs('reports', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# SESSION STATE UNTUK INTRO
if 'show_intro' not in st.session_state:
    st.session_state.show_intro = True
if 'intro_complete' not in st.session_state:
    st.session_state.intro_complete = False

# HALAMAN INTRO
if st.session_state.show_intro:
    # Hide sidebar di halaman intro
    st.markdown("""
    <style>
    [data-testid="stSidebar"] {display: none;}
    </style>
    """, unsafe_allow_html=True)
    
    # Container intro dengan animasi
    intro_col1, intro_col2, intro_col3 = st.columns([1, 2, 1])
    
    with intro_col2:
        # Logo/Icon animasi
        st.markdown("""
        <div style='text-align: center; padding: 50px;'>
            <div style='font-size: 100px; animation: bounce 2s infinite;'>🏦</div>
            <h1 style='color: #2C5282; font-size: 3rem; margin: 20px 0; 
                       animation: fadeIn 1.5s ease-in-out;'>
                CUSTOMER SENTIMENT DASHBOARD
            </h1>
            <p style='color: #94A3B8; font-size: 1.3rem; margin-bottom: 40px;'>
                Analisis Sentimen Mobile Banking Indonesia<br>
                Powered by IndoBERT & Machine Learning
            </p>
        </div>
        
        <style>
        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-20px); }
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Progress bar animasi
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simulasi loading
        for i in range(100):
            progress_bar.progress(i + 1)
            if i < 30:
                status_text.text("⚙️ Loading data...")
            elif i < 60:
                status_text.text("🤖 Initializing AI model...")
            elif i < 90:
                status_text.text("📊 Preparing visualizations...")
            else:
                status_text.text("✅ Ready!")
            time.sleep(0.02)  # Adjust speed
        
        status_text.text("🎉 Dashboard Ready!")
        st.balloons()  # Animasi balon!
        
        # Tombol masuk dashboard
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 MASUK DASHBOARD", use_container_width=True, type="primary"):
            st.session_state.show_intro = False
            st.session_state.intro_complete = True
            st.rerun()
    
    st.stop()  # Stop execution, jangan tampilkan dashboard

# 1. CONFIGURATION
print("="*80)
print("🤖 STEP 2: INDOBERT MODEL TRAINING")
print("="*80)

MODEL_NAME = "indobenchmark/indobert-base-p1"
NUM_LABELS = 3
MAX_LENGTH = 128

# OPTIMIZED BATCH SIZE UNTUK RTX 2050 4GB
BATCH_SIZE = 4  # Jangan lebih dari 4!
NUM_EPOCHS = 3  # Cukup 3 epoch

print(f"\n📦 Model: {MODEL_NAME}")
print(f"🏷️  Labels: {NUM_LABELS}")
print(f"📝 Max Length: {MAX_LENGTH}")
print(f"📦 Batch Size: {BATCH_SIZE}")
print(f"🔄 Epochs: {NUM_EPOCHS}")
print(f"💾 GPU Memory: 4.29 GB")
print(f"⚠️  Using optimized settings for 4GB VRAM")

# 2. LOAD DATA
print("\n" + "="*80)
print("📂 LOADING DATA")
print("="*80)

train_df = pd.read_csv('data/processed/train_data.csv')
test_df = pd.read_csv('data/processed/test_data.csv')

print(f"\n✅ Train samples: {len(train_df)}")
print(f"✅ Test samples: {len(test_df)}")

# 3. TOKENIZATION
print("\n" + "="*80)
print("🔤 TOKENIZATION")
print("="*80)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_data(examples):
    # Extract text column and ensure all values are strings
    texts = [str(text) if pd.notna(text) else "" for text in examples['teks_bersih']]
    
    return tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=MAX_LENGTH
    )

print("🔄 Tokenizing training data...")
train_encodings = tokenize_data(train_df)

print("🔄 Tokenizing testing data...")
test_encodings = tokenize_data(test_df)

print("✅ Tokenization complete!")

# 4. CREATE PYTORCH DATASET
class ReviewDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

print("\n" + "="*80)
print("📦 CREATING DATASETS")
print("="*80)

train_dataset = ReviewDataset(
    train_encodings, 
    train_df['label_code'].tolist()
)
test_dataset = ReviewDataset(
    test_encodings, 
    test_df['label_code'].tolist()
)

print(f"✅ Train dataset: {len(train_dataset)} samples")
print(f"✅ Test dataset: {len(test_dataset)} samples")

# 5. LOAD MODEL & FORCE GPU
print("\n" + "="*80)
print("🔧 LOADING MODEL & CHECKING GPU")
print("="*80)

# Hard-check CUDA: Langsung error jika GPU tidak terdeteksi
if not torch.cuda.is_available():
    raise RuntimeError("❌ CUDA TIDAK TERDETEKSI!\nPastikan: \n1. Driver NVIDIA terinstall & updated\n2. PyTorch versi CUDA sudah diinstall (pip install torch --index-url https://download.pytorch.org/whl/cu118)\n3. Jalankan di environment yang benar.")

device = torch.device('cuda')
print(f"🖥️  Device: {device} ({torch.cuda.get_device_name(0)})")
print(f" VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=NUM_LABELS
)

# ⚠️ KRUSIAL: Pindahkan model ke GPU secara eksplisit
model.to(device)
print("✅ Model berhasil dipindahkan ke GPU")

# 6. COMPUTE METRICS
def compute_metrics(eval_pred):
    """
    Compute evaluation metrics for business reporting
    """
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    
    # Business-critical metrics
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1_weighted': f1_score(labels, predictions, average='weighted'),
        'f1_macro': f1_score(labels, predictions, average='macro'),
        'precision_negative': precision_score(labels, predictions, average='macro', labels=[0]),
        'recall_negative': recall_score(labels, predictions, average='macro', labels=[0]),
        'f1_negative': f1_score(labels, predictions, average='macro', labels=[0]),
    }

# 7. TRAINING ARGUMENTS
print("\n" + "="*80)
print("⚙️  TRAINING CONFIGURATION")
print("="*80)

training_args = TrainingArguments(
    output_dir='./models/indobert_banking_model',
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,  
    per_device_eval_batch_size=BATCH_SIZE*2,  
    gradient_accumulation_steps=4,  
    warmup_steps=500,
    weight_decay=0.01,
    eval_strategy="epoch",  
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",
    logging_dir='./logs',
    logging_steps=50,
    
    # GPU OPTIMIZATIONS - Sudah pasti True karena ada hard-check di atas
    fp16=True,  
    dataloader_num_workers=0,   
    dataloader_prefetch_factor=None,
    
    # Memory optimization
    gradient_checkpointing=False,  
)

# 8. TRAINER
print("\n" + "="*80)
print("🚀 STARTING TRAINING")
print("="*80)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# Train the model
print("⏳ This may take 10-30 minutes depending on your GPU...")
train_result = trainer.train()

# 9. EVALUATION
print("\n" + "="*80)
print("📊 MODEL EVALUATION")
print("="*80)

eval_result = trainer.evaluate()

print("\n📈 EVALUATION METRICS:")
print("-"*60)
for key, value in eval_result.items():
    print(f"{key}: {value:.4f}")

# 10. DETAILED CLASSIFICATION REPORT
print("\n" + "="*80)
print("📋 CLASSIFICATION REPORT")
print("="*80)

predictions = trainer.predict(test_dataset)
pred_labels = torch.argmax(torch.tensor(predictions.predictions), dim=-1)
true_labels = test_df['label_code'].values

report = classification_report(
    true_labels, 
    pred_labels, 
    target_names=['Negative', 'Neutral', 'Positive'],
    output_dict=True
)

print(classification_report(
    true_labels, 
    pred_labels, 
    target_names=['Negative', 'Neutral', 'Positive']
))

# 11. CONFUSION MATRIX
print("\n" + "="*80)
print("🎯 CONFUSION MATRIX")
print("="*80)

cm = confusion_matrix(true_labels, pred_labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Neutral', 'Positive'],
            yticklabels=['Negative', 'Neutral', 'Positive'])
plt.title('Confusion Matrix - IndoBERT Sentiment Model', fontsize=14, fontweight='bold')
plt.ylabel('Actual Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.savefig('reports/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 12. SAVE MODEL & TOKENIZER
print("\n" + "="*80)
print("💾 SAVING MODEL")
print("="*80)

trainer.save_model('./models/indobert_banking_model')
tokenizer.save_pretrained('./models/indobert_banking_model')

print("✅ Model saved to: ./models/indobert_banking_model")

# 13. SAVE EVALUATION METRICS
metrics_df = pd.DataFrame({
    'Metric': list(eval_result.keys()),
    'Score': list(eval_result.values())
})
metrics_df.to_csv('reports/model_metrics.csv', index=False)

# Save classification report
report_df = pd.DataFrame(report).transpose()
report_df.to_csv('reports/classification_report.csv')

print("✅ Metrics saved to: reports/model_metrics.csv")
print("✅ Classification report saved to: reports/classification_report.csv")

# 14. BUSINESS INTERPRETATION
print("\n" + "="*80)
print("💼 BUSINESS INTERPRETATION")
print("="*80)

f1_negative = report['Negative']['f1-score']
recall_negative = report['Negative']['recall']

print(f"\n🎯 Key Business Metrics:")
print(f"   • F1-Score (Negative): {f1_negative:.2f}")
print(f"   • Recall (Negative): {recall_negative:.2f}")

if recall_negative >= 0.85:
    print("   ✅ EXCELLENT: Model detects 85%+ of customer complaints")
elif recall_negative >= 0.75:
    print("   ✅ GOOD: Model detects 75%+ of customer complaints")
else:
    print("   ⚠️  NEEDS IMPROVEMENT: Consider more training data")

print("\n" + "="*80)
print("✅ STEP 2 COMPLETE - Model Training Finished!")
print("="*80)