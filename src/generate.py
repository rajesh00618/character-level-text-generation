import os
import sys
import argparse
import json
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_lstm import LSTMModel
from src.model_transformer import TransformerModel


def load_model(model_type, model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    args_dict = checkpoint.get("args", {})
    char_to_int = checkpoint["char_to_int"]
    int_to_char = {int(k): v for k, v in checkpoint["int_to_char"].items()}
    vocab_size = checkpoint["vocab_size"]
    seq_length = args_dict.get("seq_length", 100)

    if model_type == "lstm":
        model = LSTMModel(
            vocab_size=vocab_size,
            embedding_dim=args_dict.get("embedding_dim", 128),
            hidden_dim=args_dict.get("hidden_dim", 256),
            n_layers=args_dict.get("n_layers", 2)
        ).to(device)
    else:
        model = TransformerModel(
            vocab_size=vocab_size,
            d_model=args_dict.get("hidden_dim", 128),
            n_heads=args_dict.get("n_heads", 4),
            n_layers=args_dict.get("n_layers", 2)
        ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, char_to_int, int_to_char, vocab_size, seq_length


def generate_text(model, seed_text, char_to_int, int_to_char, gen_length,
                  temperature=1.0, device="cpu", model_type="lstm", seq_length=100):
    model.eval()
    chars = list(seed_text)
    vocab_size = len(char_to_int)

    for _ in range(gen_length):
        encoded = [char_to_int.get(ch, 0) for ch in chars]
        context = encoded[-(seq_length):]
        input_seq = torch.tensor(context, dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            if model_type == "lstm":
                output, _ = model(input_seq)
            else:
                output, _ = model(input_seq)

        logits = output[-1, :] if output.dim() > 1 else output

        scaled_logits = logits / temperature
        probs = F.softmax(scaled_logits, dim=-1)
        next_char_idx = torch.multinomial(probs, 1).item()
        next_char = int_to_char[next_char_idx]
        chars.append(next_char)

    return "".join(chars)


def generate_samples(model, char_to_int, int_to_char, seed_text,
                     temperatures, gen_length, device, model_type, seq_length=100):
    samples = {}
    for temp in temperatures:
        key = f"temperature_{temp}"
        samples[key] = []
        for i in range(2):
            sample = generate_text(
                model, seed_text, char_to_int, int_to_char,
                gen_length, temperature=temp, device=device, model_type=model_type,
                seq_length=seq_length
            )
            samples[key].append(sample)
    return samples


def main():
    parser = argparse.ArgumentParser(description="Generate text using a trained model")
    parser.add_argument("--model", type=str, required=True, choices=["lstm", "transformer"],
                        help="Model type")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to the saved model checkpoint")
    parser.add_argument("--seed_text", type=str, default="To be or not to be",
                        help="Seed text to start generation")
    parser.add_argument("--gen_length", type=int, default=500,
                        help="Number of characters to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for sampling (0.5, 1.0, 1.5, etc.)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_path is None:
        args.model_path = f"models/{args.model}_model.pth"

    print(f"Loading {args.model} model from {args.model_path}...")
    model, char_to_int, int_to_char, vocab_size, seq_length = load_model(
        args.model, args.model_path, device
    )
    print(f"Model loaded. Vocabulary size: {vocab_size}, Sequence length: {seq_length}")

    generated = generate_text(
        model, args.seed_text, char_to_int, int_to_char,
        args.gen_length, temperature=args.temperature, device=device,
        model_type=args.model, seq_length=seq_length
    )

    print(f"\nSeed text: {args.seed_text}")
    print(f"Temperature: {args.temperature}")
    print(f"\nGenerated text:\n{generated}\n")


if __name__ == "__main__":
    main()
