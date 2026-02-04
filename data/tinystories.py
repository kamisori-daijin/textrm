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
from trm_build import RMSNorm, TransformerBlock, apply_rotary_pos_emb, RotaryEmbedding

class TinyStoriesDataset(Dataset):
    """Dataset for TinyStories"""
    def __init__(self, tokenizer, split='train', max_length=256, max_samples=None):
        print(f"Loading TinyStories {split} split...")
        dataset = load_dataset('roneneldan/TinyStories', split=split)

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = dataset['text']
        self.vocab_size = tokenizer.vocab_size
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        print(f"Loaded {len(self.texts)} samples")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # Add BOS/EOS handling
        tokens = self.tokenizer.encode(text, truncation=True, max_length=self.max_length)

        # Ensure all tokens are within valid range
        tokens = [min(max(t, 0), self.vocab_size - 1) for t in tokens]

        # Pad if necessary
        if len(tokens) < self.max_length:
            tokens = tokens + [self.pad_token_id] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]

        tokens = torch.tensor(tokens, dtype=torch.long)

        # Input is tokens[:-1], target is tokens[1:]
        input_ids = tokens[:-1].clone()
        targets = tokens[1:].clone()

        # Mask padding in targets (set to -100 to ignore in loss)
        targets[targets == self.pad_token_id] = -100

        return input_ids, targets
