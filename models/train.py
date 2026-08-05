import os
import sys
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from models.architecture import MultiTaskPhoBERT
from models.dataset import FeedbackDataset
from models.preprocessor import TextPreprocessor

def train_loop():
    print(f"🚀 Initializing Training Pipeline (Device: {Config.DEVICE})...")
    
    # ==========================================
    # BƯỚC 1: TẢI VÀ TIỀN XỬ LÝ DỮ LIỆU
    # ==========================================
    print("Fetching UIT-VSFC dataset from Hugging Face...")
    dataset = load_dataset("uitnlp/vietnamese_students_feedback", revision="refs/convert/parquet")
    
    train_df = pd.DataFrame(dataset['train'])
    val_df = pd.DataFrame(dataset['validation'])
    
    preprocessor = TextPreprocessor()
    
    tqdm.pandas(desc="Preprocessing Train Set")
    train_df['clean_text'] = train_df['sentence'].progress_apply(lambda x: preprocessor.clean_text(str(x)))
    
    tqdm.pandas(desc="Preprocessing Val Set")
    val_df['clean_text'] = val_df['sentence'].progress_apply(lambda x: preprocessor.clean_text(str(x)))
    
    # ==========================================
    # BƯỚC 2: TÍNH CLASS WEIGHTS
    # ==========================================
    print("Computing Class Weights for Imbalanced Data...")
    
    topic_weights = compute_class_weight(
        class_weight="balanced", 
        classes=np.unique(train_df['topic']), 
        y=train_df['topic']
    )
    tensor_topic_weights = torch.tensor(topic_weights, dtype=torch.float32).to(Config.DEVICE)

    sentiment_weights = compute_class_weight(
        class_weight="balanced", 
        classes=np.unique(train_df['sentiment']), 
        y=train_df['sentiment']
    )
    tensor_sentiment_weights = torch.tensor(sentiment_weights, dtype=torch.float32).to(Config.DEVICE)
    
    # ==========================================
    # BƯỚC 3: ĐÓNG GÓI DATALOADER
    # ==========================================
    print("Preparing DataLoaders...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    
    train_dataset = FeedbackDataset(
        train_df['clean_text'].values, train_df['topic'].values, train_df['sentiment'].values, 
        tokenizer, Config.MAX_LEN
    )
    val_dataset = FeedbackDataset(
        val_df['clean_text'].values, val_df['topic'].values, val_df['sentiment'].values, 
        tokenizer, Config.MAX_LEN
    )
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # ==========================================
    # BƯỚC 4: KHỞI TẠO MÔ HÌNH & TỐI ƯU HÓA
    # ==========================================
    print("Initializing MultiTaskPhoBERT & Optimizer...")
    model = MultiTaskPhoBERT(Config.MODEL_NAME, Config.DROPOUT).to(Config.DEVICE)
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    
    # Loss Function
    criterion_topic = nn.CrossEntropyLoss(weight=tensor_topic_weights)
    criterion_sent = nn.CrossEntropyLoss(weight=tensor_sentiment_weights)
    
    best_val_f1 = 0.0

    # ==========================================
    # BƯỚC 5: TRAIN
    # ==========================================
    for epoch in range(Config.EPOCHS):
        print(f"\n{'='*15} EPOCH {epoch+1}/{Config.EPOCHS} {'='*15}")
        
        # --- TRAINING ---
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(Config.DEVICE)
            attention_mask = batch['attention_mask'].to(Config.DEVICE)
            topics = batch['topic'].to(Config.DEVICE)
            sents = batch['sentiment'].to(Config.DEVICE)
            
            topic_logits, sent_logits = model(input_ids, attention_mask)
            
            loss = criterion_topic(topic_logits, topics) + criterion_sent(sent_logits, sents)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # --- VALIDATION ---
        model.eval()
        val_topic_preds, val_topic_trues = [], []
        val_sent_preds, val_sent_trues = [], []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(Config.DEVICE)
                attention_mask = batch['attention_mask'].to(Config.DEVICE)
                
                topic_logits, sent_logits = model(input_ids, attention_mask)
                
                val_topic_preds.extend(torch.argmax(topic_logits, dim=1).cpu().numpy())
                val_topic_trues.extend(batch['topic'].numpy())
                val_sent_preds.extend(torch.argmax(sent_logits, dim=1).cpu().numpy())
                val_sent_trues.extend(batch['sentiment'].numpy())
                
        topic_f1 = f1_score(val_topic_trues, val_topic_preds, average='macro')
        sent_f1 = f1_score(val_sent_trues, val_sent_preds, average='macro')
        combined_f1 = (topic_f1 + sent_f1) / 2
        
        print(f"Train Loss: {train_loss/len(train_loader):.4f} | Val Topic F1: {topic_f1:.4f} | Val Sent F1: {sent_f1:.4f}")
        
        # Checkpoint
        if combined_f1 > best_val_f1:
            best_val_f1 = combined_f1
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"Saved Best Model Checkpoint (F1: {combined_f1:.4f})")

if __name__ == "__main__":
    train_loop()