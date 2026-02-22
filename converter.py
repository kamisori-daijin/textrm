import torch
import torch.nn as nn
import numpy as np
import coremltools as ct
import argparse
from models.trm_model import TinyRecursiveModel
import numpy as np




# ==========================================
# 1. Setup Argument Parser
# ==========================================
parser = argparse.ArgumentParser(description="Convert TinyRecursiveModel to CoreML")
parser.add_argument('--model-path', type=str, required=True, help="Path to the trained .pt model file")

args = parser.parse_args()

# ==========================================
# 2. Build and Load Model
# ==========================================
print("[1/4] Building model...")
model = TinyRecursiveModel(
    vocab_size=50257,
    dim=256,
    n_heads=8,
    n_layers=2,
    mlp_ratio=4,
    max_seq_len=128,
    n_latent_recursions=4,
    n_improvement_cycles=2,
)

print(f"[2/4] Loading and converting weights from: {args.model_path}")
state_dict = torch.load(args.model_path, map_location="cpu")
#if 'model_state_dict' in state_dict:
    #state_dict = state_dict['model_state_dict']
#model.load_state_dict(state_dict)

model.eval()

# ==========================================
# 3. Create Dummy Input and Trace
# ==========================================
print("[3/4] Tracing model with JIT...")
example_input = torch.randint(0, 50257, (1, 128), dtype=torch.int32)

with torch.no_grad():
    traced_model = torch.jit.trace(model, (example_input,))

# ==========================================
# 4. CoreML Conversion
# ==========================================
print("[4/4] Converting to CoreML...")
mlmodel = ct.convert(
    traced_model,
    inputs=[
        ct.TensorType(
            name="input_ids",
            shape=example_input.shape,
            dtype=np.int32,
        )
    ],
    outputs=[
        ct.TensorType(name="logits")
    ],
    compute_precision=ct.precision.FLOAT16,
    compute_units=ct.ComputeUnit.CPU_AND_NE,
    minimum_deployment_target=ct.target.iOS18,
    convert_to="mlprogram",
)

# ==========================================
# 5. Add Metadata and Save
# ==========================================
mlmodel.short_description = "Tiny Recursive Model optimized for CoreML"
mlmodel.save('wikipedia-TRM.mlpackage')

print(f"\n✓ Conversion complete! ")




