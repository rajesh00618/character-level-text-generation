# Character-Level Text Generation with PyTorch

Build, train, and compare LSTM and Transformer models for character-level text generation.

## Project Structure

```
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── requirements.txt
├── README.md
├── input/
│   └── shakespeare.txt
├── results/
│   ├── loss_curves.png
│   ├── generated_samples.json
│   └── comparison_report.md
├── src/
│   ├── __init__.py
│   ├── prepare_data.py
│   ├── model_lstm.py
│   ├── model_transformer.py
│   ├── train.py
│   ├── generate.py
│   └── evaluate.py
└── models/
```

## Setup

### With Docker (Recommended)

```bash
# Build the Docker image
docker-compose build

# Download the Shakespeare dataset
docker-compose run --rm app python -c "
import urllib.request
url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
urllib.request.urlretrieve(url, 'input/shakespeare.txt')
print('Dataset downloaded')
"

# Train LSTM model
docker-compose run --rm app python src/train.py --model lstm

# Train Transformer model
docker-compose run --rm app python src/train.py --model transformer

# Full evaluation pipeline
docker-compose run --rm app python src/evaluate.py

# Generate text with LSTM
docker-compose run --rm app python src/generate.py --model lstm --temperature 1.0

# Generate text with Transformer
docker-compose run --rm app python src/generate.py --model transformer --temperature 1.0
```

### Without Docker

```bash
pip install -r requirements.txt
python src/train.py --model lstm
python src/train.py --model transformer
python src/evaluate.py
python src/generate.py --model lstm --temperature 1.0
```

## Command Line Arguments

### train.py
- `--model`: Model type (`lstm` or `transformer`)
- `--data_path`: Path to text dataset (default: `input/shakespeare.txt`)
- `--seq_length`: Sequence length (default: 100)
- `--batch_size`: Batch size (default: 64)
- `--epochs`: Number of epochs (default: 10)
- `--lr`: Learning rate (default: 0.001)

### generate.py
- `--model`: Model type (`lstm` or `transformer`)
- `--model_path`: Path to saved model checkpoint
- `--seed_text`: Seed text for generation
- `--gen_length`: Number of characters to generate
- `--temperature`: Sampling temperature (default: 1.0)
