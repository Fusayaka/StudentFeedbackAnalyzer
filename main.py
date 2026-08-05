import argparse
import json
import os
import sys
import subprocess
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from models.train import train_loop
from models.evaluate import evaluate_model
from models.inference import FeedbackPipeline

def start_ecosystem():
    """Hàm khởi chạy đồng thời cả Backend (FastAPI) và Frontend (Streamlit)"""
    backend_process = None
    frontend_process = None
    
    try:
        # 1. Bật Backend
        print("[1/2] Đang gọi API Server...")
        backend_process = subprocess.Popen([sys.executable, "api/main.py"])
        
        time.sleep(3)
        
        # 2. Bật Frontend
        print("[2/2] Đang gọi Giao diện UI...")
        frontend_process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "ui/app.py"])
        
        backend_process.wait()
        frontend_process.wait()

    except KeyboardInterrupt:
        print("\n🛑 Nhận lệnh dừng! Đang tắt toàn bộ hệ thống...")
        if backend_process:
            backend_process.terminate()
        if frontend_process:
            frontend_process.terminate()
        print("Đã tắt API Server và Giao diện UI!")

def main():
    parser = argparse.ArgumentParser(
        description="HỆ THỐNG PHÂN LOẠI FEEDBACK SINH VIÊN",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "action",
        choices=["train", "eval", "predict", "start"],
        help="Hành động bạn muốn thực thi:\n"
             "  - train   : Tải data và huấn luyện mô hình\n"
             "  - eval    : Tải trọng số tốt nhất và xuất báo cáo\n"
             "  - predict : Dự đoán nhanh một câu feedback trên terminal\n"
             "  - start   : Bật giao diện Web và API Server\n"
    )

    parser.add_argument(
        "--text",
        type=str,
        help="Văn bản cần dự đoán (Predict mode only)"
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        choices=["vncore", "underthesea"],
        default="vncore",
        help="Chọn bộ tách từ (mặc định: vncore)"
    )

    args = parser.parse_args()

    # ==========================================
    # 3. ĐIỀU PHỐI LOGIC
    # ==========================================
    if args.action == "train":
        train_loop()

    elif args.action == "eval":
        evaluate_model(Config.MODEL_SAVE_PATH)

    elif args.action == "predict":
        if not args.text:
            print("❌ [Error] Bạn phải cung cấp câu văn bản để dự đoán!")
            print("👉 Cú pháp chuẩn: python main.py predict --text \"Mạng dạo này chậm quá\"")
            return

        pipeline = FeedbackPipeline(Config.MODEL_SAVE_PATH, tokenizer_type=args.tokenizer)
        result = pipeline.predict(args.text)

        print("\n" + "="*50)
        print("🎯 KẾT QUẢ DỰ ĐOÁN")
        print("="*50)
        print(json.dumps(result, indent=4, ensure_ascii=False))
        
    elif args.action == "start":
        start_ecosystem()

if __name__ == "__main__":
    main()