import os
import sys
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from models.architecture import MultiTaskPhoBERT
from models.dataset import FeedbackDataset
from models.preprocessor import TextPreprocessor

def evaluate_model(weight_path: str = Config.MODEL_SAVE_PATH):
    print(f"Initializing Evaluation Pipeline (Device: {Config.DEVICE})...")
    print(f"Loading weights from: {weight_path}")
    
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"[Error] Không tìm thấy file trọng số tại {weight_path}. Hãy chắc chắn bạn đã train xong!")

    # ==========================================
    # BƯỚC 1: TẢI VÀ TIỀN XỬ LÝ TẬP TEST
    # ==========================================
    print("Fetching Test Set from Hugging Face...")
    dataset = load_dataset("uitnlp/vietnamese_students_feedback", revision="refs/convert/parquet")
    test_df = pd.DataFrame(dataset['test'])
    
    preprocessor = TextPreprocessor()
    
    tqdm.pandas(desc="Preprocessing Test Set")
    test_df['clean_text'] = test_df['sentence'].progress_apply(lambda x: preprocessor.clean_text(str(x)))
    
    # ==========================================
    # BƯỚC 2: CHUẨN BỊ DATALOADER & MODEL
    # ==========================================
    print("\nPreparing Model & DataLoader...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    
    test_dataset = FeedbackDataset(
        test_df['clean_text'].values, test_df['topic'].values, test_df['sentiment'].values, 
        tokenizer, Config.MAX_LEN
    )
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    model = MultiTaskPhoBERT(Config.MODEL_NAME, Config.DROPOUT)
    model.load_state_dict(torch.load(weight_path, map_location=Config.DEVICE))
    model.to(Config.DEVICE)
    model.eval()

    # ==========================================
    # BƯỚC 3: CHẠY DỰ ĐOÁN
    # ==========================================
    print("\nRunning Inference on Test Set...")
    
    topic_preds, topic_trues = [], []
    sent_preds, sent_trues = [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(Config.DEVICE)
            attention_mask = batch['attention_mask'].to(Config.DEVICE)
            
            topic_logits, sent_logits = model(input_ids, attention_mask)
            
            topic_preds.extend(torch.argmax(topic_logits, dim=1).cpu().numpy())
            topic_trues.extend(batch['topic'].numpy())
            
            sent_preds.extend(torch.argmax(sent_logits, dim=1).cpu().numpy())
            sent_trues.extend(batch['sentiment'].numpy())

    # ==========================================
    # IN CLASSIFICATION REPORT
    # ==========================================
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    
    topic_target_names = [Config.TOPIC_MAPPING[i] for i in range(len(Config.TOPIC_MAPPING))]
    sent_target_names = [Config.SENT_MAPPING[i] for i in range(len(Config.SENT_MAPPING))]

    print("\n1. Topic")
    print(f"Accuracy: {accuracy_score(topic_trues, topic_preds):.4f}")
    print(classification_report(topic_trues, topic_preds, target_names=topic_target_names, zero_division=0))
    
    print("\n2. Sentiment")
    print(f"Accuracy: {accuracy_score(sent_trues, sent_preds):.4f}")
    print(classification_report(sent_trues, sent_preds, target_names=sent_target_names, zero_division=0))

if __name__ == "__main__":
    evaluate_model()