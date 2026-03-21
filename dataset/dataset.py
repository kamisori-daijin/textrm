import torch
from torch.utils.data import Dataset
from datasets import load_dataset

class Dataset(Dataset):
    def __init__(
        self,
        tokenizer,
        dataset_name="Kamisori-daijin/email-datasets-20k",
        max_length=128,
        max_samples=20000,
        split="train",
        val_split=False,
        val_split_ratio=0.9
    ):
        print(f"Loading CoT dataset from {dataset_name}...")
        
        # load streaming
        dataset = load_dataset(
            dataset_name,
            split=split,
            streaming=True
        )

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        buffer = []

        for item in dataset:
            
            
            text = item.get("text", "")
            
            if not text:
                continue

            # 2. tokenize
           
            tokens = tokenizer.encode(text)
            
           
            buffer.extend(tokens)

            # 4. Packing process (128 destinations + 1 item for next forecast)
            # Slide using buffer[max_length:] to extract without duplicates.
            while len(buffer) >= (max_length + 1):
                chunk = buffer[:max_length + 1]
                self.examples.append(torch.tensor(chunk))
                buffer = buffer[max_length:] # Place 1 token on top and move on to the next step.

            # The program will end once the specified number of samples has been reached.
            if len(self.examples) >= max_samples:
                break

        # valdation split
        if val_split:
            split_idx = int(len(self.examples) * val_split_ratio)
            self.examples = self.examples[split_idx:]

        print(f"Built {len(self.examples)} samples (Sequence Length: {max_length}).")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens = self.examples[idx]

        # Shift input_ids by one position, like this: [0, 1, 2] -> target: [1, 2, 3]
        input_ids = tokens[:-1].clone()
        targets = tokens[1:].clone()

        return input_ids, targets