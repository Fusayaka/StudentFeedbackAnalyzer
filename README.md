# Student Feedback Classifier (UIT-VSFC)

Multi-task Vietnamese NLP system that classifies student feedback along two axes simultaneously:

- **Topic**: Giảng viên (Lecturer) / Chương trình đào tạo (Curriculum) / Cơ sở vật chất (Facility) / Khác (Other)
- **Sentiment**: Tiêu cực (Negative) / Trung tính (Neutral) / Tích cực (Positive)

Built on a fine-tuned [PhoBERT](https://huggingface.co/vinai/phobert-base) with two classification heads, trained on the [UIT-VSFC](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback) dataset. Served through a FastAPI backend with a Streamlit UI on top.

## Project Structure

```
api/            FastAPI backend (serves /predict, /health)
ui/             Streamlit frontend
models/         Model architecture, dataset, tokenizer, preprocessing, train/eval logic
data/           Teencode normalization dictionary
weights/        Trained model checkpoint (best.pt) — not tracked in git
VnCoreNLP/      Word segmenter, auto-downloaded on first run — not tracked in git
config.py       Central configuration (paths, hyperparameters, label mappings)
main.py         CLI entry point (train / eval / predict / start)
```

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and fill in your values:
   ```
   HF_TOKEN=your_huggingface_token_here
   JAVA_HOME="C:\path\to\your\jdk"
   ```
   `JAVA_HOME` is required on Windows to run the VnCoreNLP word segmenter (needs a JDK installed separately).

## Usage

All commands run through `main.py`:

```bash
# Train the model on UIT-VSFC
python main.py train

# Evaluate the best checkpoint and print a classification report
python main.py eval

# Quick single-sentence prediction from the terminal
python main.py predict --text "Thầy giảng bài rất hay, nhưng phòng máy lạnh bị hỏng nóng quá"

# Launch the API server (FastAPI) and web UI (Streamlit) together
python main.py start
```

Use `--tokenizer vncore` (default) or `--tokenizer underthesea` to choose the Vietnamese word segmenter for `predict`.

Once running, the UI is available at `http://localhost:8501` and the API at `http://localhost:8000`.

## Notes

- Model weights (`weights/best.pt`) are not committed — run `python main.py train` to produce one locally.
- `VnCoreNLP/` is downloaded automatically on first use and is not committed.
