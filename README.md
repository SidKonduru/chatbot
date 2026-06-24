# Seq2Seq Chatbot

An encoder-decoder LSTM chatbot built with TensorFlow 2 / Keras.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1. Preprocess raw data (optional)

If you have `data/chat.txt` (alternating question/answer lines):

```bash
python main.py preprocess
```

The project already includes preprocessed `idx_q.npy`, `idx_a.npy`, and `metadata.pkl`, so you can skip this step unless you want to rebuild from source.

### 2. Train

```bash
python main.py train
```

Options:

- `--epochs 50` — max training epochs (default: 50)
- `--batch-size 64` — batch size (default: 64)
- `--emb-dim 512` — embedding size (default: 512)
- `--num-layers 2` — LSTM layers (default: 2)
- `--patience 5` — early stopping patience (default: 5)

Checkpoints are saved to `weights/best_model.keras`.

### 3. Chat

```bash
python main.py chat
```

Type `quit` or `exit` to stop.

## Project layout

| File | Purpose |
|------|---------|
| `main.py` | CLI: `preprocess`, `train`, `chat` |
| `data_preprocessing.py` | Raw text → indexed arrays |
| `data_utils.py` | Split, encode/decode, batch helpers |
| `seq2seq_model.py` | TF 2 Keras seq2seq model |
| `data/chat.txt` | Raw Q/A dataset (not included) |
| `weights/` | Trained model checkpoints |

## Data format

`data/chat.txt` should contain one utterance per line, alternating question then answer:

```
hi there
hello how can i help you
what is your name
i am a chatbot
```
