import torch
from models.birefnet import BiRefNet

# paths
ckpt = "epoch_400.pth"
onnx_path = "birefnet.onnx"

# init model
model = BiRefNet()
state_dict = torch.load(ckpt, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# dummy input
dummy = torch.randn(1, 3, 512, 512)  # adjust size to what you use

torch.onnx.export(
    model,
    dummy,
    "birefnet.onnx",
    export_params=True,
    input_names=['input'],
    output_names=['output'],
    opset_version=18,
    dynamic_axes=None,   # <- REMOVE dynamic_axes for OpenCV compatibility
)


print(f"Exported ONNX -> {onnx_path}")
