import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from torch.utils.data import DataLoader

from config import Config
from models.dataset import FeedbackDataset
from models.preprocessor import TextPreprocessor

def load_uit_vsfc():
    return load_dataset(Config.UIT_VSFC_REPO, revision=Config.UIT_VSFC_REVISION)


def preprocess_split(raw_dataset, split: str, preprocessor: TextPreprocessor = None) -> pd.DataFrame:
    df = pd.DataFrame(raw_dataset[split])
    preprocessor = preprocessor or TextPreprocessor()

    tqdm.pandas(desc=f"Preprocessing {split.capitalize()} Set")
    df['clean_text'] = df['sentence'].progress_apply(lambda x: preprocessor.clean_text(str(x)))
    return df


def build_dataloader(df: pd.DataFrame, tokenizer, batch_size: int = Config.BATCH_SIZE, shuffle: bool = False) -> DataLoader:
    dataset = FeedbackDataset(
        df['clean_text'].values, df['topic'].values, df['sentiment'].values,
        tokenizer, Config.MAX_LEN
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
