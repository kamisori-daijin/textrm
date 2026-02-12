import torch
import argparse
import coremltools as ct
import torch
from models.trm_model import TinyRecursiveModel


parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str, required=True, help='Path to the model file to load')

args = parser.parse_args()

model = torch.load(args.model_path)

def convert_model(self, model: TinyRecursiveModel) -> ct.models.MLModel:
    """Convert TRM model to CoreML format"""
    traced_model = torch.jit.trace(model)
    mlmodel = ct.convert (
        traced_model,
        inputs=[ct.TensorType(name="input", shape=traced_model.input_shapes[0])],
        outputs=[ct.TensorType(name="output", shape=traced_model.output_shapes[0])],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )
    return mlmodel



model.eval()
print("Model loaded and set to eval mode.")

mlmodel = convert_model(model)    
mlmodel.save("trm_model.mlpackage")
print("CoreML model saved as trm_model.mlpackage")



