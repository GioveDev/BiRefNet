import onnx
onnx.checker.check_model("birefnet.onnx")
print("ONNX OK")
