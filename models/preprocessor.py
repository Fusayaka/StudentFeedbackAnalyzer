import re
import json
import os
from models.tokenizer import Tokenizer

class TextPreprocessor:
    def __init__(self, dict_path: str = "teencode.json", tokenizer = None):
        self.tokenizer = tokenizer
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        self.dict_path = os.path.join(project_root, "data", dict_path)
        
        self.teencode_dict = self._load_dictionary()

    def _load_dictionary(self) -> dict:
        if not os.path.exists(self.dict_path):
            print(f"[Lưu ý] Không tìm thấy file {os.path.basename(self.dict_path)}. Hệ thống sẽ bỏ qua bước chuẩn hóa teencode.")
            return {}
            
        try:
            with open(self.dict_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[Error] Dữ liệu JSON bị hỏng hoặc sai format ({e}). Tạm thời bỏ qua từ điển.")
            return {}

    def _remove_noise(self, text):
        text = str(text).lower()
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[^\w\s_áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ\.,!?]', ' ', text) 
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _normalize_teencode(self, text):
        words = text.split()
        normalized_words = [self.teencode_dict.get(w, w) for w in words]
        return ' '.join(normalized_words)

    def _handle_lengthening(self, text):
        return re.sub(r'([a-z_áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ])\1+', r'\1', text)

    def _segment_words(self, text):
        if self.tokenizer:
            return self.tokenizer.tokenize(text)
        return text

    def clean_text(self, text):
        if not text or not isinstance(text, str):
            return ""
        text = self._remove_noise(text)
        text = self._handle_lengthening(text)
        text = self._normalize_teencode(text)
        text = self._segment_words(text)
        return text