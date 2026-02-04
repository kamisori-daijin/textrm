from contextlib import contextmanager
from functools import partial

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
from ema.ema import EMA

#@contextmanager

def train(
    model,
    train_loader,
    val_loader,
    tokenizer,
    device,
    epochs=5,
    lr=1e-4,
    warmup_steps=1000,
    n_supervision_steps=4,
    ema_decay=0.999,
    save_path='trm_tinystories.pt'
):
    """Training loop with deep supervision and EMA"""

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    ema = EMA(model, decay=ema_decay)

    # Learning rate scheduler with warmup
    def lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    global_step = 0
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')

        for input_ids, targets in pbar:
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            loss = model(input_ids, targets, n_supervision_steps=n_supervision_steps)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            ema.update()

            global_step += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.6f}'})

        # Validation
        ema.apply_shadow()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for input_ids, targets in tqdm(val_loader, desc='Validation'):
                input_ids = input_ids.to(device)
                targets = targets.to(device)
                loss = model(input_ids, targets, n_supervision_steps=n_supervision_steps)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f'Epoch {epoch+1} - Val Loss: {val_loss:.4f}')

        # Generate sample
        prompt = "Once upon a time"
        prompt_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
        generated = model.generate(prompt_ids, max_new_tokens=100)
        generated_text = tokenizer.decode(generated[0].tolist())
        print(f'Generated: {generated_text[:300]}...\n')

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'ema_shadow': ema.shadow,
                'epoch': epoch,
                'val_loss': val_loss
            }, save_path)
            print(f'Saved best model with val_loss={val_loss:.4f}')

        ema.restore()

    return model