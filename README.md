# PyTorch Tiny Recursive Models (TRM)

A simplified and efficient reimplementation of [TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels), optimized for memory-constrained environments and modern training techniques.

Hugging Face:
- [Kamisori-daijin/textrm-28M-bizmail](https://huggingface.co/Kamisori-daijin/textrm-28M-bizmail)
- [Kamisori-daijin/textrm1.5-25M-bizmail](https://huggingface.co/Kamisori-daijin/textrm1.5-25M-bizmail)



## Key Features

- **Recursive Latent Reasoning**: Implements the core TRM architecture where a single "tiny" network is reused across latent recursions (`n`) and improvement cycles (`T`).
- **Deep Supervision**: Trains with intermediate losses across multiple refinement steps to ensure stable convergence.
- **Efficient Packing Strategy**: 
    - Multiple training examples are packed into fixed-length blocks (e.g., 256 tokens) with `</s>` separators.
    - Zero padding: maximizes GPU throughput by ensuring every token in the batch is used for learning.
- **Smart Loss Masking**: 
    - The dataset automatically identifies the `<user>` prompt and masks it during training using `-100`.
    - The model focuses exclusively on learning the reasoning process (`<think>...</think>`) and final output (`<generate>...</generate>`).
- **Optimized Data Pipeline**: 
    - Single-pass dataset loading and splitting from streaming sources.
    - Reduced memory footprint and initialization time.

## Usage

### 1. Setup the Environment

```bash
python -m venv .venv
source .venv/bin/activate  # MacOS/Linux
# or .venv\Scripts\activate on Windows
pip install -r requirements.txt 
```

### 2. Configure the Model

Adjust hyperparameters in `models/config.py`. The defaults are tuned for stability on consumer hardware (e.g., Apple Silicon M-series).

```python
config = {
    "vocab_size": 32005,      # TinyLlama(32k) + Special Tokens
    "dim": 512,               # Hidden dimension
    "max_seq_len": 256,       # Sequence length
    "n_latent_recursions": 4, # 'n' in the TRM paper
    "n_improvement_cycles": 2,# 'T' in the TRM paper
    "batch_size": 16,
    "n_supervision_steps": 3, # Number of supervision steps
}
```

### 3. Training

Launch the training script. It will automatically download the dataset (streaming), pack it, and begin training with EMA (Exponential Moving Average) weights.

```bash
python train.py 
```

### 4. Weights & Inference

Convert the PyTorch checkpoints to Safetensors for better compatibility:

```bash
python convert_to_safetensors.py
```

Run a simple generation test:

```bash
python inference.py
```

## Dataset Format

The training pipeline expects a specific format to enable efficient masking:

```json
{
  "text": "<user>Prompt Here</user><think>Reasoning steps...</think><generate>Final output...</generate></s>"
}
```
*Note: The model learns to generate everything after the `<user>` section.*

## Acknowledgments

- [SamsungSAILMontreal/TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) - Original research and implementation.
- [gmarchetti2020/TRM-Experiments](https://github.com/gmarchetti2020/TRM-Experiments) - Training insights.
- [stockeh/mlx-trm](https://github.com/stockeh/mlx-trm) - Project structure inspiration.

---
Created by Kamisori-daijin
