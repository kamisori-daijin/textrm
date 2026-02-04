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


class TinyRecursiveNetwork(nn.Module):
    """
    The core tiny network used in TRM.
    Only 2 layers as per the paper's finding that smaller is better.
    """
    def __init__(self, dim, n_heads=8, n_layers=2, mlp_ratio=4, max_seq_len=512):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, mlp_ratio, max_seq_len)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class TinyRecursiveModel(nn.Module):
    """
    Tiny Recursive Model for Text Generation

    Architecture based on TRM paper:
    - Single tiny 2-layer network
    - Recursive reasoning with latent z and prediction y
    - Deep supervision across multiple improvement steps

    For text generation:
    - x: embedded input sequence (context)
    - y: current token predictions (embedded)
    - z: latent reasoning state

    The model recursively improves its latent z, then updates y.
    """
    def __init__(
        self,
        vocab_size,
        dim=256,
        n_heads=8,
        n_layers=2,
        mlp_ratio=4,
        max_seq_len=256,
        n_latent_recursions=6,  # n in the paper
        n_improvement_cycles=3,  # T in the paper
    ):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.n_latent_recursions = n_latent_recursions
        self.n_improvement_cycles = n_improvement_cycles

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        # Single tiny network (key insight: one network is better than two)
        self.net = TinyRecursiveNetwork(dim, n_heads, n_layers, mlp_ratio, max_seq_len)

        # Projection layers for combining x, y, z
        self.combine_xyz = nn.Linear(dim * 3, dim, bias=False)
        self.combine_yz = nn.Linear(dim * 2, dim, bias=False)

        # Output head
        self.output_head = nn.Linear(dim, vocab_size, bias=False)

        # Halting head for ACT (simplified - no Q-learning)
        self.halt_head = nn.Linear(dim, 1, bias=False)

        # Learnable initial states for y and z
        self.y_init = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.z_init = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_embeddings(self, input_ids):
        """Get token + position embeddings"""
        B, T = input_ids.shape
        # Clamp input_ids to valid range
        input_ids = input_ids.clamp(0, self.vocab_size - 1)
        # Clamp position to max_seq_len
        T = min(T, self.max_seq_len)
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        return self.token_emb(input_ids[:, :T]) + self.pos_emb(pos)

    def latent_recursion(self, x, y, z):
        """
        Single recursion cycle:
        1. Update z n times given (x, y, z)
        2. Update y once given (y, z)
        """
        # Latent reasoning: update z n times
        for _ in range(self.n_latent_recursions):
            combined = self.combine_xyz(torch.cat([x, y, z], dim=-1))
            z = self.net(combined)

        # Refine prediction: update y given (y, z)
        combined_yz = self.combine_yz(torch.cat([y, z], dim=-1))
        y = self.net(combined_yz)

        return y, z

    def deep_recursion(self, x, y, z, use_grad=True):
        """
        Deep recursion with T improvement cycles.
        First T-1 cycles without gradients, last cycle with gradients.
        """
        if not use_grad:
            # All cycles without gradients (inference)
            with torch.no_grad():
                for _ in range(self.n_improvement_cycles):
                    y, z = self.latent_recursion(x, y, z)
            return y.detach(), z.detach()

        # T-1 cycles without gradients
        with torch.no_grad():
            for _ in range(self.n_improvement_cycles - 1):
                y, z = self.latent_recursion(x, y, z)

        # Last cycle with gradients
        y, z = self.latent_recursion(x, y, z)

        return y.detach(), z.detach(), self.output_head(y), self.halt_head(y.mean(dim=1))

    def forward(self, input_ids, targets=None, n_supervision_steps=4):
        """
        Forward pass with deep supervision.

        Args:
            input_ids: [B, T] input token IDs
            targets: [B, T] target token IDs (for training)
            n_supervision_steps: number of deep supervision steps

        Returns:
            If training: loss
            If inference: logits
        """
        B, T = input_ids.shape
        T = min(T, self.max_seq_len)
        input_ids = input_ids[:, :T]

        x = self.get_embeddings(input_ids)

        # Initialize y and z
        y = self.y_init.expand(B, T, -1).clone()
        z = self.z_init.expand(B, T, -1).clone()

        if targets is None:
            # Inference: just run deep recursion
            y, z = self.deep_recursion(x, y, z, use_grad=False)
            return self.output_head(y)

        # Ensure targets match input length
        targets = targets[:, :T]

        # Training with deep supervision
        total_loss = 0.0

        for step in range(n_supervision_steps):
            y, z, logits, halt_logit = self.deep_recursion(x, y, z, use_grad=True)

            # Cross-entropy loss for token prediction
            ce_loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=-100
            )

            # Halting loss (simplified ACT)
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                mask = (targets != -100)
                correct = ((preds == targets) & mask).float().sum() / mask.float().sum().clamp(min=1)
            halt_loss = F.binary_cross_entropy_with_logits(
                halt_logit.squeeze(-1),
                correct.expand(B)
            )

            total_loss = total_loss + ce_loss + 0.1 * halt_loss

        return total_loss / n_supervision_steps

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=0.8, top_k=40):
        """Generate text autoregressively"""
        self.eval()

        for _ in range(max_new_tokens):
            # Crop to max_seq_len - 1 to leave room for prediction
            idx_cond = input_ids[:, -(self.max_seq_len - 1):]

            # Clamp input ids to valid vocab range
            idx_cond = idx_cond.clamp(0, self.vocab_size - 1)

            # Get predictions
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

