import torch
import torch.nn as nn
import numpy as np
import coremltools as ct
import argparse
from models.trm_model import TinyRecursiveModel
import numpy as np

# ==========================================
# Helper: Convert old state_dict to new format
# ==========================================
def convert_state_dict(old_state_dict):
    """
    Convert state_dict from Linear-based architecture to Conv2d-based architecture.
    
    Transformations:
    1. qkv.weight -> [q_proj, k_proj, v_proj] weights
    2. proj.weight -> o_proj.weight  
    3. Linear weights -> Conv2d weights (reshape to 4D)
    4. RMSNorm weights remain unchanged
    """
    new_state_dict = {}
    
    for key, value in old_state_dict.items():
        # Skip RMSNorm weights - they're compatible
        if 'norm' in key and 'weight' in key:
            new_state_dict[key] = value
            continue
        
        # Convert RotaryEmbedding buffers (no change needed)
        if 'rope' in key and 'weight' not in key:
            new_state_dict[key] = value
            continue
        
        # Convert Attention layers: qkv -> q_proj, k_proj, v_proj
        if 'attn.qkv.weight' in key:
            # Old: [3*dim, dim], New: 3x [dim, dim, 1, 1]
            dim = value.shape[1]
            q_weight, k_weight, v_weight = value.split(dim, dim=0)
            
            new_key = key.replace('attn.qkv.weight', 'attn.q_proj.weight')
            new_state_dict[new_key] = q_weight.reshape(dim, dim, 1, 1)
            
            new_key = key.replace('attn.qkv.weight', 'attn.k_proj.weight')
            new_state_dict[new_key] = k_weight.reshape(dim, dim, 1, 1)
            
            new_key = key.replace('attn.qkv.weight', 'attn.v_proj.weight')
            new_state_dict[new_key] = v_weight.reshape(dim, dim, 1, 1)
            continue
        
        # Convert Attention output projection: proj -> o_proj
        if 'attn.proj.weight' in key:
            new_key = key.replace('attn.proj.weight', 'attn.o_proj.weight')
            # Linear [dim, dim] -> Conv2d [dim, dim, 1, 1]
            new_state_dict[new_key] = value.reshape(value.shape[0], value.shape[1], 1, 1)
            continue
        
        # Convert MLP weights: Linear -> Conv2d
        if 'mlp.w' in key and 'weight' in key:
            # Linear [out_features, in_features] -> Conv2d [out_features, in_features, 1, 1]
            new_state_dict[key] = value.reshape(value.shape[0], value.shape[1], 1, 1)
            continue
        
        # Copy all other keys (combine_xyz, combine_yz, output_head, halt_head, embeddings, etc.)
        if 'weight' in key or 'bias' in key:
            new_state_dict[key] = value
        elif 'y_init' in key or 'z_init' in key:
            new_state_dict[key] = value
        else:
            # Parameters or buffers
            new_state_dict[key] = value
    
    return new_state_dict


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
checkpoint = torch.load(args.model_path, map_location="cpu")

# Extract model state dict from checkpoint if needed
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    old_state_dict = checkpoint['model_state_dict']
else:
    old_state_dict = checkpoint

# Convert old state_dict to new format
print("     Converting state_dict format (Linear -> Conv2d)...")
new_state_dict = convert_state_dict(old_state_dict)

# Load converted state dict
try:
    model.load_state_dict(new_state_dict, strict=False)
    print("     ✓ State dict loaded successfully")
except RuntimeError as e:
    print(f"     Warning: {e}")
    print("     Attempting to load with strict=False...")
    model.load_state_dict(new_state_dict, strict=False)

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
mlmodel.save(args.output)

print(f"\n✓ Conversion complete! Saved to: {args.output}")




