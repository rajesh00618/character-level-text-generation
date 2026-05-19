import os
import torch
from torch.utils.data import Dataset, DataLoader


class CharDataset(Dataset):
    def __init__(self, text, seq_length):
        self.text = text
        self.seq_length = seq_length

    def __len__(self):
        return max(0, len(self.text) - self.seq_length)

    def __getitem__(self, idx):
        x = self.text[idx : idx + self.seq_length]
        y = self.text[idx + 1 : idx + self.seq_length + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def load_text(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    return text


def create_vocab(text):
    chars = sorted(list(set(text)))
    char_to_int = {ch: i for i, ch in enumerate(chars)}
    int_to_char = {i: ch for i, ch in enumerate(chars)}
    return chars, char_to_int, int_to_char


def encode_text(text, char_to_int):
    return [char_to_int[ch] for ch in text]


def get_train_test_split(data, train_ratio=0.9):
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]


def prepare_dataloaders(text, seq_length, batch_size, char_to_int, train_ratio=0.9):
    encoded = encode_text(text, char_to_int)
    train_data, test_data = get_train_test_split(encoded, train_ratio)

    train_dataset = CharDataset(train_data, seq_length)
    test_dataset = CharDataset(test_data, seq_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, test_loader


def get_encoded_text_and_vocab(filepath):
    text = load_text(filepath)
    chars, char_to_int, int_to_char = create_vocab(text)
    encoded = encode_text(text, char_to_int)
    return encoded, chars, char_to_int, int_to_char


def get_dataloaders(filepath, seq_length, batch_size, train_ratio=0.9):
    text = load_text(filepath)
    chars, char_to_int, int_to_char = create_vocab(text)
    train_loader, test_loader = prepare_dataloaders(text, seq_length, batch_size, char_to_int, train_ratio)
    return train_loader, test_loader, chars, char_to_int, int_to_char
