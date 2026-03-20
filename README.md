# pytorch Tiny Recursive Models 

Simplified reimplementation of [TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)

## Usage

1. Setup the environment

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install requirements library
   ```bash
   pip install -r requirements.txt 
   ```

2. Adjust model config in `models/config.py`

   ```python
    config = {
      'vocab_size': 50257,  # GPT-2 vocab
      'dim': 256,           # Hidden dimension
      'n_heads': 8,         # Attention heads
      'n_layers': 2,        # Only 2 layers (key insight from paper)
      'mlp_ratio': 4,
      'max_seq_len': 128,   # Reduced for stability
      'n_latent_recursions': 4,  # n in paper (reduced for memory)
      'n_improvement_cycles': 2,  # T in paper (reduced for memory)

      # Training
      'batch_size': 64,     # Reduced for MPS memory constraints
      'epochs': 15,
      'lr': 1e-4,
      'warmup_steps': 500,
      'n_supervision_steps': 3,  # Deep supervision steps during training
      'max_train_samples': 20000,  # Reduced for memory and speed
      'max_val_samples': 1000,
    }
   ```
3. Train on Email Datasets:
   ```bash
   python train.py 
  
   ```
4. Convert weight to Safetensors
  ```bash
  python convert_to_saferensors.py
  ```
5. Run Inference
 ```bash
 python inference.py
 ```

## Thanks

[gmarchetti2020/TRM-Experiments](https://github.com/gmarchetti2020/TRM-Experiments)-Defining and training the model

[stockeh/mlx-trm](https://github.com/stockeh/mlx-trm)-Tool structure


