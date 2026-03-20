import torch
from transformers import GPT2Tokenizer
from models.trm_model import TinyRecursiveModel
from safetensors.torch import load_file
from models.config import config

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

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
state_dict = load_file("final_model.safetensors")
model.load_state_dict(state_dict)
model.to(device)
model.eval()


def generate_email(prompt, max_new_tokens=150, temperature=0.8):
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    generated = model.generate(prompt_ids, max_new_tokens=max_new_tokens, temperature=temperature)
    return tokenizer.decode(generated[0].tolist())


if __name__ == "__main__":
    prompts = [
        "Write a polite refusal email",
    ]

    print("\n=== Generated Emails ===\n")
    for prompt in prompts:
        email = generate_email(prompt)
        print(f'Prompt: "{prompt}"')
        print(f"Email: {email}\n")
        print("-" * 50 + "\n")
