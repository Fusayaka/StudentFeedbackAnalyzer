import os
import sys
from sklearn.metrics import classification_report, accuracy_score
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from models.data import load_uit_vsfc, preprocess_split, build_dataloader
from models.utils import build_model, run_inference_loop

def evaluate_model(weight_path: str = Config.MODEL_SAVE_PATH):
    print(f"Initializing Evaluation Pipeline (Device: {Config.DEVICE})...")
    print(f"Loading weights from: {weight_path}")

    if not os.path.exists(weight_path):
        raise FileNotFoundError(
            f"[Error] Không tìm thấy file trọng số tại {weight_path}. Hãy chắc chắn bạn đã train xong!"
        )

    # ==========================================
    # BƯỚC 1: TẢI VÀ TIỀN XỬ LÝ TẬP TEST
    # ==========================================
    print("Fetching Test Set from Hugging Face...")
    raw_dataset = load_uit_vsfc()
    test_df = preprocess_split(raw_dataset, 'test')

    # ==========================================
    # BƯỚC 2: CHUẨN BỊ DATALOADER & MODEL
    # ==========================================
    print("\nPreparing Model & DataLoader...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    test_loader = build_dataloader(test_df, tokenizer, Config.BATCH_SIZE, shuffle=False)

    model = build_model(weight_path)

    # ==========================================
    # BƯỚC 3: CHẠY DỰ ĐOÁN
    # ==========================================
    print("\nRunning Inference on Test Set...")
    topic_preds, topic_trues, sent_preds, sent_trues = run_inference_loop(model, test_loader)

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
