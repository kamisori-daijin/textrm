import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer
import math
import os
from tqdm import tqdm
import copy
from models import trm_build
from models.trm_build import RMSNorm, TransformerBlock, apply_rotary_pos_emb, RotaryEmbedding

class WikipediaDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        max_length=128,
        max_samples=100000,
        split="train"
    ):
        print("Loading Wikipedia...")
        dataset = load_dataset(
            "wikipedia",
            "20220301.en",
            split=split,
            streaming=True
        )

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        buffer = []

        for i, item in enumerate(dataset):
            text = item["text"]

        
            if len(text) < 200:
                continue

            tokens = tokenizer.encode(text)

            buffer.extend(tokens)

        
            while len(buffer) >= max_length:
                chunk = buffer[:max_length]
                buffer = buffer[max_length:]
                self.examples.append(torch.tensor(chunk))

            if len(self.examples) >= max_samples:
                break

        print(f"Built {len(self.examples)} samples.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens = self.examples[idx]

        input_ids = tokens[:-1].clone()
        targets = tokens[1:].clone()

        return input_ids, targets  
