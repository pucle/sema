"""
CODE CHẠY LOCAL - HOÀN TOÀN MIỄN PHÍ
API Key chỉ dùng để tải model lần đầu, sau đó chạy offline được
"""

from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
import cv2
import time
import onnxruntime as ort
import os

# MONKEY PATCH: Force OpenVINO to use GPU
OriginalSession = ort.InferenceSession

class PatchedSession(OriginalSession):
    def __init__(self, path_or_bytes, **kwargs):
        providers = kwargs.get("providers", [])
        if not providers:
            # inference might set check providers later or pass them differently
            pass
        
        # Check if we should inject OpenVINO options
        new_providers = []
        for p in providers:
            if p == "OpenVINOExecutionProvider" or (isinstance(p, tuple) and p[0] == "OpenVINOExecutionProvider"):
                print("DEBUG: Patching OpenVINO to use GPU_FP16")
                # Use tuple format (name, options)
                new_providers.append(("OpenVINOExecutionProvider", {"device_type": "GPU_FP16"}))
            else:
                new_providers.append(p)
        
        if new_providers:
            kwargs["providers"] = new_providers
            
        super().__init__(path_or_bytes, **kwargs)

ort.InferenceSession = PatchedSession

# Check available providers
print(f"DEBUG: ONNX Runtime Providers: {ort.get_available_providers()}")
# Force inference lib to select OpenVINO
if "OpenVINOExecutionProvider" in ort.get_available_providers():
    print("DEBUG: OpenVINO detected. Setting env var.")
    os.environ["ROBOFLOW_INFERENCE_DEVICE"] = "openvino"


class LocalVideoProcessor:
    def __init__(self):
        self.frame_count = 0
        self.start_time = time.time()
        self.detection_count = 0
        
    def process_predictions(self, predictions: dict, video_frame: VideoFrame):
        """Xử lý predictions từ model - CHẠY HOÀN TOÀN LOCAL"""
        
        self.frame_count += 1
        frame = video_frame.image.copy()
        
        # Tính FPS
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        # Vẽ predictions lên frame
        if predictions and "predictions" in predictions:
            for pred in predictions["predictions"]:
                self.detection_count += 1
                
                # Tính toán bounding box
                x = int(pred["x"] - pred["width"] / 2)
                y = int(pred["y"] - pred["height"] / 2)
                w = int(pred["width"])
                h = int(pred["height"])
                confidence = pred["confidence"]
                class_name = pred["class"]
                
                # Vẽ box màu xanh lá
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Vẽ label với background
                label = f"{class_name}: {confidence:.2%}"
                (label_w, label_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(frame, (x, y - label_h - 10), 
                            (x + label_w, y), (0, 255, 0), -1)
                cv2.putText(frame, label, (x, y - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Hiển thị thông tin FPS và detections
        info_text = f"FPS: {fps:.1f} | Frames: {self.frame_count} | Detections: {len(predictions.get('predictions', []))}"
        cv2.putText(frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        
        # Hiển thị frame
        cv2.imshow("Roboflow Local Detection", frame)
        
        # Nhấn 'q' để thoát
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print(f"\n✅ Đã xử lý {self.frame_count} frames")
            print(f"📊 Tổng detections: {self.detection_count}")
            print(f"⚡ FPS trung bình: {fps:.2f}")
            pipeline.terminate()
        
        # In log mỗi 30 frames
        if self.frame_count % 30 == 0:
            print(f"Frame {self.frame_count} | FPS: {fps:.1f} | "
                  f"Detections: {len(predictions.get('predictions', []))}")

# Khởi tạo processor
processor = LocalVideoProcessor()

# QUAN TRỌNG: Tìm workspace_name và model_id của bạn
# Truy cập: https://app.roboflow.com/
# Chọn project -> Settings -> copy "workspace/project/version"
# Ví dụ: "trai26/vehicle-detection/3"

print("="*60)
print("🚀 ROBOFLOW LOCAL INFERENCE - MIỄN PHÍ 100%")
print("="*60)
print("📌 API Key chỉ dùng để tải model lần đầu")
print("📌 Sau đó chạy hoàn toàn LOCAL, không tốn credit")
print("📌 Nhấn 'q' để thoát")
print("="*60)

# Khởi tạo pipeline - CHẠY LOCAL
pipeline = InferencePipeline.init(
    api_key="ylFu6Gi5msSoDxbPC9Sl",  # API key của bạn
    model_id="semaphore-dataset-1wlaa/1",  # ⚠️ THAY ĐỔI: workspace/project/version
    video_reference=0,  # 0 = webcam, hoặc đường dẫn video
    max_fps=30,  # Giới hạn FPS để tránh quá tải
    on_prediction=processor.process_predictions,
)

try:
    pipeline.start()
    pipeline.join()
except KeyboardInterrupt:
    print("\n⛔ Dừng bởi người dùng")
finally:
    cv2.destroyAllWindows()
    print("\n✅ Đã đóng tất cả cửa sổ")