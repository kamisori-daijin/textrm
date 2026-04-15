config = {
    "vocab_size": 32005,  # TinyLlama(32k) + 5 Special Tokens
    "dim": 512,  # Hidden dimension
    "n_heads": 16,  # Attention heads
    "n_layers": 3,  # Only 3 layers (key insight from paper)
    "mlp_ratio": 4,
    "max_seq_len": 256,  # Reduced for stability
    "n_latent_recursions": 4,  # n in paper (reduced for memory)
    "n_improvement_cycles": 2,  # T in paper (reduced for memory)
    # Training
    "batch_size": 16,  # Reduced for MPS memory constraints
    "gradient_accumulation_steps": 4,
    "epochs": 20,
    "lr": 1e-4,
    "warmup_steps": 500,
    "n_supervision_steps": 3,  # Deep supervision steps during training
    "max_train_samples": 100000,  # Reduced for memory and speed
    "max_val_samples": 1000,
}
