import torch
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Env
    HF_TOKEN = os.getenv("HF_TOKEN")
    JAVA_HOME = os.getenv("JAVA_HOME")

    # System
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths
    VNCORE_DIR = "./VnCoreNLP"
    MODEL_SAVE_PATH = os.path.join(ROOT_DIR, "weights", "best.pt")
    
    # Model & Training Parameters
    MODEL_NAME = "vinai/phobert-base"
    MAX_LEN = 40
    BATCH_SIZE = 32
    EPOCHS = 5
    LEARNING_RATE = 2e-5
    DROPOUT = 0.3
    
    # Label Mappings
    TOPIC_MAPPING = {
        0: "Giảng viên", 
        1: "Chương trình đào tạo", 
        2: "Cơ sở vật chất", 
        3: "Khác"
    }
    
    SENT_MAPPING = {
        0: "Tiêu cực", 
        1: "Trung tính", 
        2: "Tích cực"
    }