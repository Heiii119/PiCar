import time
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

interpreter = Interpreter(
    model_path="model_edgetpu.tflite",
    experimental_delegates=[load_delegate("libedgetpu.so.1")]
)

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']

dummy = np.zeros(input_shape, dtype=np.uint8)

start = time.time()

for _ in range(100):
    interpreter.set_tensor(input_details[0]['index'], dummy)
    interpreter.invoke()
    interpreter.get_tensor(output_details[0]['index'])

end = time.time()

print("Average inference time:", (end - start)/100, "seconds")
