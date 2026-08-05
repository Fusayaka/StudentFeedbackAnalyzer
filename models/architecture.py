import torch.nn as nn
from transformers import AutoModel

class MultiTaskPhoBERT(nn.Module):
    def __init__(self, model_name, dropout):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        
        # 4 topics, 3 sentiments
        self.topic_head = nn.Linear(768, 4)
        self.sentiment_head = nn.Linear(768, 3)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)

        topic_logits = self.topic_head(pooled_output)
        sent_logits = self.sentiment_head(pooled_output)

        return topic_logits, sent_logits