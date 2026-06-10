
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re

class SentimentPredictor:
    def __init__(self, model_path='./models/indobert_banking_model'):
        """Load model dan tokenizer"""
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            # Load model dan tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            
            # Label mapping (sesuai training Anda)
            self.label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
            
            self.loaded = True
            print(f"✅ Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.loaded = False
    
    def clean_text(self, text):
        """Preprocessing teks sama seperti saat training"""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+', '', text)  # Hapus URL
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Hapus simbol
        text = ' '.join(text.split())  # Hapus spasi berlebih
        return text
    
    def predict(self, text):
        """Prediksi sentimen dari satu teks"""
        if not self.loaded:
            return {'error': 'Model not loaded'}
        
        # Preprocessing
        cleaned_text = self.clean_text(text)
        
        # Tokenize
        inputs = self.tokenizer(
            cleaned_text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        ).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        return {
            'sentiment': self.label_mapping[prediction],
            'confidence': confidence,
            'probabilities': {
                'negative': probabilities[0][0].item(),
                'neutral': probabilities[0][1].item(),
                'positive': probabilities[0][2].item()
            },
            'original_text': text,
            'cleaned_text': cleaned_text
        }