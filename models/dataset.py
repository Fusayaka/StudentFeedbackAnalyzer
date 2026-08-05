import torch
from torch.utils.data import Dataset

class FeedbackDataset(Dataset):
    def __init__(self, texts, topics, sentiments, tokenizer, max_len):
        self.texts = texts
        self.topics = topics
        self.sentiments = sentiments
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'topic': torch.tensor(self.topics[item], dtype=torch.long),
            'sentiment': torch.tensor(self.sentiments[item], dtype=torch.long)
        }