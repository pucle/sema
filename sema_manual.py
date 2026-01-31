import cv2
import time
import os
import onnxruntime as ort
from inference import get_model

# 1. MONKEY PATCH: Force OpenVINO to use GPU
OriginalSession = ort.InferenceSession

class PatchedSession(OriginalSession):
    def __init__(self, path_or_bytes, **kwargs):
        providers = kwargs.get("providers", [])
        
        # Check if we should inject OpenVINO options
        new_providers = []
        for p in providers:
            if p == "OpenVINOExecutionProvider" or (isinstance(p, tuple) and p[0] == "OpenVINOExecutionProvider"):
                print("DEBUG: Patching OpenVINO to use GPU_FP16")
                new_providers.append(("OpenVINOExecutionProvider", {"device_type": "GPU_FP16"}))
            else:
                new_providers.append(p)
        
        if new_providers:
            kwargs["providers"] = new_providers
            
        super().__init__(path_or_bytes, **kwargs)

ort.InferenceSession = PatchedSession

# Force env var just in case
if "OpenVINOExecutionProvider" in ort.get_available_providers():
    os.environ["ROBOFLOW_INFERENCE_DEVICE"] = "openvino"

# 2. Config
API_KEY = "ylFu6Gi5msSoDxbPC9Sl"
MODEL_ID = "semaphore-dataset-1wlaa/1"

print("Loading model...")
model = get_model(model_id=MODEL_ID, api_key=API_KEY)
print("Model loaded.")

# 3. Camera config
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)

frame_count = 0
start_time = time.time()

print("Starting loop...")
while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame_count += 1
    
    # Inference
    results = model.infer(frame)[0] # get_model inference returns list of results? assuming single frame
    
    # Visualization
    predictions = results.predictions
    
    for pred in predictions:
        # Check predict structure (it might be object or dict depending on version)
        # Using getattr or dict access
        if hasattr(pred, "x"):
            x, y, w, h = int(pred.x), int(pred.y), int(pred.width), int(pred.height)
            class_name = pred.class_name
            confidence = pred.confidence
        else:
             # Fallback for dict
            x = int(pred["x"] - pred["width"] / 2)
            y = int(pred["y"] - pred["height"] / 2)
            w = int(pred["width"])
            h = int(pred["height"])
            class_name = pred["class"]
            confidence = pred["confidence"]

        # Draw
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{class_name}: {confidence:.2%}"
        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # FPS Calculation
    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0
    
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("Manual Inference", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
