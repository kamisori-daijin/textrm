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
from models.trm_model import TinyRecursiveModel
from models.config import config



device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {device}')

# Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Model
model = TinyRecursiveModel(
    vocab_size=config['vocab_size'],
    dim=config['dim'],
    n_heads=config['n_heads'],
    n_layers=config['n_layers'],
    mlp_ratio=config['mlp_ratio'],
    max_seq_len=config['max_seq_len'],
    n_latent_recursions=config['n_latent_recursions'],
    n_improvement_cycles=config['n_improvement_cycles'],
)

# Count parameters
n_params = sum(p.numel() for p in model.parameters())
print(f'Model parameters: {n_params:,} ({n_params/1e6:.2f}M)')
print(f'Effective depth per supervision step: {config["n_improvement_cycles"] * (config["n_latent_recursions"] + 1) * config["n_layers"]}')
     
