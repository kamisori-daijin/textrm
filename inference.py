import torch
from transformers import AutoTokenizer
from models.trm_model import TinyRecursiveModel
from safetensors.torch import load_file
from models.config import config

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("./textrm-2.0-tokenizer")
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


def generate_email(prompt, max_new_tokens=300, temperature=0.7):
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    generated = model.generate(prompt_ids, max_new_tokens=max_new_tokens, temperature=temperature)
    return tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)


if __name__ == "__main__":
    prompts = [
        #level 1
        "Write a polite refusal email",
        "Write a professional business email.",
        "Write a short formal email.",
        #level 2
        "Write a polite refusal email to a client.",
        "Write a polite refusal email to a partner.",
        "Write a polite refusal email to a vendor.",
        #level 3
        "Write a polite refusal email to a client regarding a budget request.",
        "Write a polite refusal email to a partner regarding a meeting request.",
        "Write a polite refusal email to a vendor regarding a contract proposal.",
        #level 4
        "Write a polite and formal refusal email.",
        "Write a polite but firm refusal email.",
        "Write a slightly apologetic refusal email.",
        #level 5
        "Write a polite refusal email under 100 words.",
        "Write a concise refusal email.",
        #level 6
        "Write a polite refusal email from a Product Manager.",
        "Write a polite refusal email from a CEO.",
        "Write a polite refusal email from a Customer Success Manager.",
        #level 7
        "Write a polite refusal email from a Customer Success Manager to a client regarding a pricing request.",
        "Write a polite refusal email from a CEO to a partner regarding a proposal.",
        "Write a polite refusal email from a Product Manager to a vendor regarding a feature request.",
        #level 8
        "Write a passive-aggressive refusal email.",
        "Write a very enthusiastic refusal email.",
        "Write a refusal email that sounds overly positive.",
        #level 9
        "Refuse a request politely in an email.",
        "Politely decline a business request via email."
    ]

    print("\n=== Generated Emails ===\n")
    for prompt in prompts:
        email = generate_email(prompt)
        print(f'Prompt: "{prompt}"')
        print(f"Email: {email}\n")
        print("-" * 50 + "\n")
