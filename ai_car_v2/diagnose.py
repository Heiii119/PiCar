# diagnose.py
import onnx
import numpy as np
import onnxruntime as ort
import cv2

# --- Fact 1: what input layout does the model expect? ---
m = onnx.load("model.onnx")
inp = m.graph.input[0]
shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
print("INPUT name:", inp.name)
print("INPUT shape:", shape)   # [1,224,224,3]=NHWC(TF) | [1,3,224,224]=NCHW

# --- Fact 2: is the output softmax (sums to ~1) or raw logits? ---
sess = ort.InferenceSession("model.onnx")
in_name = sess.get_inputs()[0].name

# build a dummy input matching whatever layout we found
if shape[1] == 3:           # NCHW
    dummy = np.random.rand(1, 3, 224, 224).astype(np.float32)
else:                       # NHWC
    dummy = np.random.rand(1, 224, 224, 3).astype(np.float32)

out = sess.run(None, {in_name: dummy})[0][0]
print("OUTPUT raw:", out)
print("OUTPUT sum:", float(np.sum(out)))   # ~1.0 => softmax; otherwise logits
print("OUTPUT len:", len(out))             # should be 6 (your class count)
