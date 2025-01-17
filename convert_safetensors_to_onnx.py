import torch
from safetensors.torch import load_file
import timm


model_path = "models/vit_large_patch14_dinov2.safetensors"
model_weights = load_file(model_path)

model = timm.create_model("vit_large_patch14_dinov2",
                          pretrained=False, features_only=True)
model.load_state_dict(model_weights, strict=False)
model.eval()

dummy_input = torch.randn(1, 3, 518, 518).to(torch.float32)
onnx_output_path = "vit_large_patch14_dinov2_float32.onnx"

torch.onnx.export(
    model,
    dummy_input,
    onnx_output_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=14,
    strict=False,
)

print(f"Exported {onnx_output_path}")
