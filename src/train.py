import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prepare_data import get_dataloaders
from src.model_lstm import LSTMModel
from src.model_transformer import TransformerModel


def train_model(model, train_loader, test_loader, criterion, optimizer, device, n_epochs, clip=1.0, model_name="lstm"):
    train_losses = []
    test_losses = []

    for epoch in range(n_epochs):
        model.train()
        total_train_loss = 0
        n_batches = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            if model_name == "lstm":
                outputs, _ = model(inputs)
            else:
                outputs, _ = model(inputs)

            loss = criterion(outputs, targets.view(-1))
            loss.backward()

            if clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()

            total_train_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_train_loss / n_batches
        train_losses.append(avg_train_loss)

        model.eval()
        total_test_loss = 0
        n_test_batches = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                if model_name == "lstm":
                    outputs, _ = model(inputs)
                else:
                    outputs, _ = model(inputs)
                loss = criterion(outputs, targets.view(-1))
                total_test_loss += loss.item()
                n_test_batches += 1

        avg_test_loss = total_test_loss / n_test_batches
        test_losses.append(avg_test_loss)

        print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")

    return train_losses, test_losses


def main():
    parser = argparse.ArgumentParser(description="Train a character-level text generation model")
    parser.add_argument("--model", type=str, required=True, choices=["lstm", "transformer"],
                        help="Model type to train: lstm or transformer")
    parser.add_argument("--data_path", type=str, default="input/shakespeare.txt",
                        help="Path to the text dataset")
    parser.add_argument("--seq_length", type=int, default=100,
                        help="Sequence length for training")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--embedding_dim", type=int, default=128,
                        help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Hidden dimension (LSTM) / d_model (Transformer)")
    parser.add_argument("--n_layers", type=int, default=2,
                        help="Number of LSTM layers or Transformer encoder layers")
    parser.add_argument("--n_heads", type=int, default=4,
                        help="Number of attention heads (Transformer only)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--clip", type=float, default=1.0,
                        help="Gradient clipping value")
    parser.add_argument("--save_dir", type=str, default="models",
                        help="Directory to save trained models")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory to save results")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    print(f"Loading data from {args.data_path}...")
    train_loader, test_loader, chars, char_to_int, int_to_char = get_dataloaders(
        args.data_path, args.seq_length, args.batch_size
    )
    vocab_size = len(chars)
    print(f"Vocabulary size: {vocab_size}")

    if args.model == "lstm":
        model = LSTMModel(
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers
        ).to(device)
        print(f"LSTM model created: embedding_dim={args.embedding_dim}, "
              f"hidden_dim={args.hidden_dim}, n_layers={args.n_layers}")
    else:
        model = TransformerModel(
            vocab_size=vocab_size,
            d_model=args.hidden_dim,
            n_heads=args.n_heads,
            n_layers=args.n_layers
        ).to(device)
        print(f"Transformer model created: d_model={args.hidden_dim}, "
              f"n_heads={args.n_heads}, n_layers={args.n_layers}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"\nStarting training for {args.epochs} epochs...")
    train_losses, test_losses = train_model(
        model, train_loader, test_loader, criterion, optimizer,
        device, args.epochs, args.clip, args.model
    )

    model_filename = f"{args.model}_model.pth"
    model_path = os.path.join(args.save_dir, model_filename)
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_size": vocab_size,
        "char_to_int": char_to_int,
        "int_to_char": int_to_char,
        "args": vars(args)
    }, model_path)
    print(f"Model saved to {model_path}")

    results = {
        "model": args.model,
        "train_losses": train_losses,
        "test_losses": test_losses,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_test_loss": test_losses[-1] if test_losses else None,
    }

    results_path = os.path.join(args.results_dir, f"{args.model}_training_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Training results saved to {results_path}")
    print("Training complete!")


if __name__ == "__main__":
    main()
