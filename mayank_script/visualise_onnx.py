# Validate exported model
import onnx
f="onnx/yolov4.onnx"
model = onnx.load(f)  # Load the ONNX model
onnx.checker.check_model(model)  # Check that the IR is well formed
print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph
# return