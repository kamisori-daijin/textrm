from datasets import load_from_disk
import torch
from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(
        self,
        dataset_path,
        tokenizer,
        max_length=128,
        max_samples=100000,
        val_split=False,
        val_split_ratio=0.9
    ):
        print("Loading local dataset...")

        dataset = load_from_disk(dataset_path)

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        buffer = []

        for i, item in enumerate(dataset):
            text = item["text"]

            tokens = tokenizer.encode(text)

            buffer.extend(tokens)

            while len(buffer) >= max_length:
                chunk = buffer[:max_length]
                buffer = buffer[max_length:]
                self.examples.append(torch.tensor(chunk))

            if len(self.examples) >= max_samples:
                break

        if val_split:
            split_idx = int(len(self.examples) * val_split_ratio)
            self.examples = self.examples[split_idx:]

        print(f"Built {len(self.examples)} samples.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens = self.examples[idx]
        input_ids = tokens[:-1].clone()
        targets = tokens[1:].clone()
        return input_ids, targets