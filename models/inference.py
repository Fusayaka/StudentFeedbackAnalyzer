import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from models.tokenizer import VnCoreNLPTokenizer, UndertheseaTokenizer
from models.preprocessor import TextPreprocessor
from models.utils import build_model

class FeedbackPipeline:
    def __init__(self, model_path: str, tokenizer_type: str = "vncore"):
        print(f"🚀 Initializing Inference Pipeline (Device: {Config.DEVICE})...")
        self.device = Config.DEVICE
        
        # ==========================================
        # 1. KHỞI TẠO TOKENIZER & PREPROCESSOR
        # ==========================================
        print(f"Loading Word Segmenter ({tokenizer_type.upper()})...")
        if tokenizer_type == "vncore":
            self.word_segmenter = VnCoreNLPTokenizer()
        elif tokenizer_type == "underthesea":
            self.word_segmenter = UndertheseaTokenizer()
        else:
            raise ValueError(f"[LỖI] tokenizer_type '{tokenizer_type}' không hợp lệ. Chọn 'vncore' hoặc 'underthesea'.")
            
        self.preprocessor = TextPreprocessor(tokenizer=self.word_segmenter)
        
        # ==========================================
        # 2. KHỞI TẠO PHOBERT TOKENIZER
        # ==========================================
        print("Loading PhoBERT Tokenizer...")
        self.bert_tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        
        # ==========================================
        # 3. NẠP TRỌNG SỐ MÔ HÌNH
        # ==========================================
        print(f"Loading model weights from: {model_path}...")
        self.model = build_model(model_path)
        self.model.eval()
        print("Model is ready for inference!")

    def predict(self, raw_text: str):
        cleaned_text = self.preprocessor.clean_text(raw_text)
        
        encoded = self.bert_tokenizer._encode_plus(
            cleaned_text,
            max_length=Config.MAX_LEN,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        with torch.no_grad():
            topic_logits, sent_logits = self.model(input_ids, attention_mask)
            
            topic_probs = F.softmax(topic_logits, dim=1).squeeze()
            sent_probs = F.softmax(sent_logits, dim=1).squeeze()
            
            topic_pred = torch.argmax(topic_probs, dim=0).item()
            sent_pred = torch.argmax(sent_probs, dim=0).item()
            
        return {
            "text": raw_text,
            "cleaned_text": cleaned_text,
            "topic": {
                "label": Config.TOPIC_MAPPING.get(topic_pred),
                "confidence": round(topic_probs[topic_pred].item() * 100, 2)
            },
            "sentiment": {
                "label": Config.SENT_MAPPING.get(sent_pred),
                "confidence": round(sent_probs[sent_pred].item() * 100, 2)
            }
        }

if __name__ == "__main__":
    pipeline = FeedbackPipeline(Config.MODEL_SAVE_PATH, tokenizer_type="vncore")
    
    test_str = "nhà vệ sinh dơ"
    result = pipeline.predict(test_str)
    
    print("\n" + "="*40)
    print("🎯 INFERENCE RESULT")
    print("="*40)
    
    import json
    print(json.dumps(result, indent=4, ensure_ascii=False))