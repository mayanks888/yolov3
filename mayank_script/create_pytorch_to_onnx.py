import torch
import torch.onnx
path="yolo_v4_mayank.pt"
path="/home/mayank_s/Downloads/yolov3.pt"
model = torch.load(path)

# dummy_input = XXX # You need to provide the correct input here!
# # dummy_input = 608 # You need to provide the correct input here!
# imgsz=412
# dummy_input = torch.zeros((1, 3, imgsz, imgsz), device="cuda") # You need to provide the correct input here!
#
# # Check it's valid by running the PyTorch model
# dummy_output = model(dummy_input)
# print("Input is valid")
#
# # If the input is valid, then exporting should work
#
# torch.onnx.export(model, dummy_input, "pytorch_model.onnx")




# torch.onnx.export(model, dummy_input, path)
#
# imgsz=(412,412)
# model.fuse()
# img = torch.zeros((1, 3) + imgsz)  # (1, 3, 320, 192)
# f = "pytorch_model.onnx"
# torch.onnx.export(model, img, f, verbose=False, opset_version=11,
#                   input_names=['images'], output_names=['classes', 'boxes'])

