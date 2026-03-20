import torch
from safetensors.torch import save_file

from models.config import config
from models.trm_model import TinyRecursiveModel

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Model
model = TinyRecursiveModel(
    vocab_size=config["vocab_size"],
    dim=config["dim"],
    n_heads=config["n_heads"],
    n_layers=config["n_layers"],
    mlp_ratio=config["mlp_ratio"],
    max_seq_len=config["max_seq_len"],
    n_latent_recursions=config["n_latent_recursions"],
    n_improvement_cycles=config["n_improvement_cycles"],
)

# Load trained weights
checkpoint = torch.load("final_model.pt", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

# Save as safetensors
save_file(model.state_dict(), "final_model.safetensors")
print("Converted final_model.pt to final_model.safetensors")
