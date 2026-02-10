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
    'epochs': 3,
    'lr': 1e-4,
    'warmup_steps': 500,
    'n_supervision_steps': 3,  # Deep supervision steps during training
    'max_train_samples': 2000000,  # Limit for faster training demo
    'max_val_samples': 20000,
}