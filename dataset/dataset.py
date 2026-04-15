import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from tqdm import tqdm

class PackedDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # examples[idx] is a dict: {"input_ids": tensor, "labels": tensor}
        ex = self.examples[idx]
        return ex["input_ids"], ex["labels"]

def get_packed_dataset(
    tokenizer,
    dataset_name="Kamisori-daijin/email-datasets-v2-100k",
    max_length=256,
    max_samples= 100000,
    split="train",
    val_ratio=0.1
):
    print(f"Loading and packing dataset: {dataset_name}")
    
    dataset = load_dataset(dataset_name, split=split, streaming=True)
    
    all_packed_examples = []
    buffer_ids = []
    buffer_labels = []
    
    # Special tokens
    think_token = "<think>"
    eos_id = tokenizer.eos_token_id

    pbar = tqdm(total=max_samples, desc="Packing data")
    
    for item in dataset:
        text = item.get("text", "")
        if not text:
            continue
        
        # Loss Masking Logic:
        # We want to mask everything before <think>
        parts = text.split(think_token)
        if len(parts) < 2:
            # If <think> is not found, we might want to skip or treat differently.
            # Here we just treat the whole thing as label.
            prompt_text = ""
            completion_text = text
        else:
            prompt_text = parts[0]
            completion_text = think_token + parts[1]

        # Tokenize prompt (to be masked)
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        # Tokenize completion (to be learned)
        completion_ids = tokenizer.encode(completion_text, add_special_tokens=False)
        if not completion_ids or completion_ids[-1] != eos_id:
            completion_ids.append(eos_id)

        # Create IDs and Labels
        item_ids = prompt_ids + completion_ids
        item_labels = ([-100] * len(prompt_ids)) + completion_ids

        buffer_ids.extend(item_ids)
        buffer_labels.extend(item_labels)

        # Packing process
        while len(buffer_ids) >= max_length:
            chunk_ids = buffer_ids[:max_length]
            chunk_labels = buffer_labels[:max_length]
            
            # Since the model predicts the NEXT token:
            # input:  tokens[0...N-1]
            # target: labels[1...N]
            # We need max_length tokens to produce (max_length-1) length sequences
            # But wait, if config['max_seq_len'] is 256, we need 257 tokens in the chunk.
            
            all_packed_examples.append({
                "input_ids": torch.tensor(chunk_ids[:-1], dtype=torch.long),
                "labels": torch.tensor(chunk_labels[1:], dtype=torch.long)
            })
            
            # Move buffer forward (non-overlapping)
            # Actually, to avoid losing the target for the last token of the chunk,
            # we should keep 1 token overlap if we want continuous prediction.
            # But for standard packing, we just slide.
            buffer_ids = buffer_ids[max_length-1:]
            buffer_labels = buffer_labels[max_length-1:]

            pbar.update(1)
            if len(all_packed_examples) >= max_samples:
                break
        
        if len(all_packed_examples) >= max_samples:
            break
            
    pbar.close()

    # Split into train and val
    val_size = int(len(all_packed_examples) * val_ratio)
    train_size = len(all_packed_examples) - val_size
    
    # Random split or sequential? Usually sequential for streaming is fine.
    train_examples = all_packed_examples[:train_size]
    val_examples = all_packed_examples[train_size:]
    
    print(f"Total blocks: {len(all_packed_examples)} (Train: {len(train_examples)}, Val: {len(val_examples)})")
    
    return PackedDataset(train_examples), PackedDataset(val_examples)
