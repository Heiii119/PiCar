from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate

try:
    interpreter = Interpreter(
        model_path="model_edgetpu.tflite",
        experimental_delegates=[load_delegate("libedgetpu.so.1")]
    )
    interpreter.allocate_tensors()
    print("✅ Edge TPU is working!")
except Exception as e:
    print("❌ Edge TPU failed:", e)
