# MLX Tiny Recursive Models

Simplified reimplementation of [TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) using [MLX](https://github.com/ml-explore/mlx).

## Usage

1. Setup the environment

   ```bash
   uv sync
   source .venv/bin/activate
   ```
2. Install requirements library
   ```bash
   pip install -r requirements.txt 
   ```

2. Adjust model config in `train.py`

   ```python
   @dataclass
   class ModelConfig:
       in_channels: int
       depth: int
       dim: int
       heads: int
       patch_size: tuple
       n_outputs: int
       pool: str = "cls" # mean or cls
       n: int = 6  # latent steps
       T: int = 3  # deep steps
       halt_max_steps: int = 8  # maximum supervision steps
       halt_exploration_prob: float = 0.2  # exploratory q probability
       halt_follow_q: bool = True  # follow q (True) or max steps (False)
   ```
3. Train on TinyStories:
   ```bash
   python train.py 
  
   ```



