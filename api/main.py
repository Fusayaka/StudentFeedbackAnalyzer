import sys
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from models.inference import FeedbackPipeline

pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    pipeline = FeedbackPipeline(Config.MODEL_SAVE_PATH)
    print("✅ Đã nạp xong! API Server sẵn sàng nhận request.\n")
    yield
    print("\n🧹 Đang giải phóng bộ nhớ và tắt Máy ảo Java...")
    pipeline = None

app = FastAPI(
    title="UIT-VSFC Feedback API",
    lifespan=lifespan
)

class FeedbackRequest(BaseModel):
    text: str

@app.get("/health")
async def health_check():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="AI is loading")
    return {"status": "ready"}

@app.post("/predict")
async def predict_feedback(request: FeedbackRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Mô hình đang khởi động, vui lòng thử lại sau vài giây.")
        
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Văn bản không được để trống")
    
    try:
        result = pipeline.predict(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý AI: {str(e)}")

if __name__ == "__main__":
    print("Đang chuẩn bị môi trường API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)