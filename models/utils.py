import os

import torch
from tqdm import tqdm

from config import Config
from models.architecture import MultiTaskPhoBERT


def build_model(weight_path: str = None) -> MultiTaskPhoBERT:
    model = MultiTaskPhoBERT(Config.MODEL_NAME, Config.DROPOUT)

    if weight_path:
        if not os.path.exists(weight_path):
            raise FileNotFoundError(
                f"[Error] Không tìm thấy file trọng số tại {weight_path}. Hãy chắc chắn bạn đã train xong!"
            )
        model.load_state_dict(torch.load(weight_path, map_location=Config.DEVICE))

    model.to(Config.DEVICE)
    return model


@torch.no_grad()
def run_inference_loop(model, loader, desc: str = "Evaluating"):
    model.eval()

    topic_preds, topic_trues = [], []
    sent_preds, sent_trues = [], []

    for batch in tqdm(loader, desc=desc):
        input_ids = batch['input_ids'].to(Config.DEVICE)
        attention_mask = batch['attention_mask'].to(Config.DEVICE)

        topic_logits, sent_logits = model(input_ids, attention_mask)

        topic_preds.extend(torch.argmax(topic_logits, dim=1).cpu().numpy())
        topic_trues.extend(batch['topic'].numpy())
        sent_preds.extend(torch.argmax(sent_logits, dim=1).cpu().numpy())
        sent_trues.extend(batch['sentiment'].numpy())

    return topic_preds, topic_trues, sent_preds, sent_trues
