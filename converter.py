import torch
import numpy as np
import coremltools as ct
import argparse
from models.trm_model import TinyRecursiveModel

# ==========================================
# 1. Setup Argument Parser
# ==========================================
parser = argparse.ArgumentParser(description="Convert TinyRecursiveModel to CoreML")
parser.add_argument('--model-path', type=str, required=True, help="Path to the trained .pt model file")
parser.add_argument('--output', type=str, default="trm_model.mlpackage", help="Output CoreML model name")
args = parser.parse_args()

# ==========================================
# 2. Build and Load Model
# ==========================================
# Make sure these hyperparameters match your trained model
model = TinyRecursiveModel(
    vocab_size=50257,
    dim=256,
    n_heads=4,
    n_layers=2,
    mlp_ratio=4,
    max_seq_len=128,
    n_latent_recursions=4,
    n_improvement_cycles=2,
)

print(f"Loading weights from: {args.model_path}")
state_dict = torch.load(args.model_path, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# ==========================================
# 3. Create Dummy Input and Trace
# ==========================================
# Input shape: (Batch Size, Sequence Length) -> (1, 128)
# Use random integers within the vocabulary range [0, 50256]
example_input = torch.randint(0, 50257, (1, 128), dtype=torch.int32)

print("Tracing model with JIT...")
# Tracing the inference path (where targets=None)
with torch.no_grad():
    traced_model = torch.jit.trace(model, (example_input,))

# ==========================================
# 4. CoreML Conversion
# ==========================================
print("Starting CoreML conversion...")
mlmodel = ct.convert(
    traced_model,
    inputs=[
        ct.TensorType(
            name="input_ids",
            shape=example_input.shape,
            dtype=torch.int32, # NLP token IDs must be integers
        )
    ],
    outputs=[
        ct.TensorType(name="logits")
    ],
    # Optimization settings for Apple Neural Engine (ANE)
    compute_precision=ct.precision.FLOAT16,
    compute_units=ct.ComputeUnit.ALL,
    minimum_deployment_target=ct.target.iOS18,
    convert_to="mlprogram",
)

# ==========================================
# 5. Add Metadata and Save
# ==========================================

mlmodel.short_description = "Tiny Recursive Model optimized for CoreML"

mlmodel.save(args.output)
print(f"Conversion complete! Saved to: {args.output}")




