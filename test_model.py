"""
TEST MODEL LOADING
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("🔍 Testing model loading...")

# Path
model_path = './models/indobert_banking_model'

try:
    # Load
    print(f"📂 Loading from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    print("✅ Model loaded successfully!")
    print(f"📊 Model config: {model.config}")
    
    # Test prediction
    test_text = "Aplikasi sangat bagus dan mudah digunakan"
    inputs = tokenizer(test_text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    
    labels = {0: 'negative', 1: 'neutral', 2: 'positive'}
    print(f"✅ Test prediction: '{test_text}'")
    print(f"   → Sentiment: {labels[pred]}")
    print(f"   → Confidence: {probs[0][pred].item()*100:.1f}%")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()