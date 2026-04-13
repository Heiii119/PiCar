import cv2
import numpy as np
import time
from picamera2 import Picamera2
from tflite_runtime.interpreter import Interpreter, load_delegate

# ====== SETTINGS ======
MODEL_PATH = "best_int8_edgetpu.tflite"
INPUT_SIZE = 320
CONF_THRESHOLD = 0.4

CLASS_NAMES = ["Line", "My-First-Project", "Stop"]  # change if needed
# ======================

# Load Coral model
interpreter = Interpreter(
    model_path=MODEL_PATH,
    experimental_delegates=[load_delegate("libedgetpu.so.1")]
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Setup Pi AI Camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

print("✅ Coral + Camera Ready")

def preprocess(frame):
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.uint8)
    return img

def postprocess(output, frame):
    h, w, _ = frame.shape
    detections = output[0]  # (1, 7, 2100)

    for det in detections.T:
        conf = det[4]
        if conf > CONF_THRESHOLD:
            class_id = int(np.argmax(det[5:]))
            score = det[4]

            cx, cy, bw, bh = det[0], det[1], det[2], det[3]

            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)

            label = f"{CLASS_NAMES[class_id]} {score:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    return frame

# FPS counter
prev_time = time.time()

while True:
    frame = picam2.capture_array()
    input_data = preprocess(frame)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    frame = postprocess(output, frame)

    # FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(frame, f"FPS: {fps:.1f}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    cv2.imshow("Coral Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
