import cv2
import time

from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters.common import input_size, set_input
from pycoral.adapters.detect import get_objects

# Load model
interpreter = make_interpreter("coco_ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite")
interpreter.allocate_tensors()

width, height = input_size(interpreter)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized = cv2.resize(frame, (width, height))
    set_input(interpreter, resized)

    start = time.time()
    interpreter.invoke()
    inference_time = time.time() - start

    objs = get_objects(interpreter, score_threshold=0.5)

    for obj in objs:
        bbox = obj.bbox
        cv2.rectangle(frame,
                      (int(bbox.xmin), int(bbox.ymin)),
                      (int(bbox.xmax), int(bbox.ymax)),
                      (0, 255, 0), 2)

    fps = 1.0 / inference_time
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2)

    cv2.imshow("Coral Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
