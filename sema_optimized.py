"""
SEMAPHORE DETECTOR - OPTIMIZED VERSION (CPU)
=============================================
- FPS cao hơn với camera optimization
- In kết quả phát hiện real-time
- Signal tracking và smoothing
- Sử dụng CPU thay vì GPU yếu
"""

import cv2
import time
import os
import onnxruntime as ort
from datetime import datetime

# Import detection modules
from detection_logger import DetectionLogger, SignalTracker

# ============================================================
# 1. CPU CONFIGURATION - Better than weak integrated GPU
# ============================================================
print("🔧 Configuring CPU inference...")
os.environ["ROBOFLOW_INFERENCE_DEVICE"] = "cpu"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
print("✅ CPU mode enabled (4 threads)")

# ============================================================
# 2. CONFIGURATION
# ============================================================

# Camera settings
CAMERA_WIDTH = 640      # Giảm từ HD -> 640 để tăng FPS
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_BUFFER = 1       # Buffer nhỏ = latency thấp

# Detection settings  
CONFIDENCE_THRESHOLD = 0.65  # Ngưỡng confidence
DEBOUNCE_TIME = 0.4          # Thời gian debounce (giây)
CONFIRM_FRAMES = 2           # Số frame để confirm signal

# Frame skip (0 = process all frames)
FRAME_SKIP = 0  # Đặt = 1 để skip mỗi frame thứ 2, = 2 để process 1/3 frames

# Logging
ENABLE_CSV_LOG = True
LOG_FILE = "detections.csv"

# Display
SHOW_SEQUENCE = True     # Hiển thị chuỗi tín hiệu
MAX_FPS_DISPLAY = 60     # Giới hạn display FPS

# ============================================================
# 3. IMPORT ROBOFLOW (after monkey patch)
# ============================================================

from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame


# ============================================================
# 4. OPTIMIZED VIDEO PROCESSOR
# ============================================================

class OptimizedVideoProcessor:
    def __init__(self):
        self.frame_count = 0
        self.process_count = 0
        self.start_time = time.time()
        self.last_frame_time = time.time()
        
        # Detection logger
        self.logger = DetectionLogger(
            debounce_time=DEBOUNCE_TIME,
            min_confidence=CONFIDENCE_THRESHOLD,
            log_file=LOG_FILE if ENABLE_CSV_LOG else None,
            print_enabled=True
        )
        
        # Signal tracker
        self.tracker = SignalTracker(
            confirm_frames=CONFIRM_FRAMES,
            min_confidence=CONFIDENCE_THRESHOLD - 0.1
        )
        
        # For FPS calculation
        self.fps_buffer = []
        self.fps_buffer_size = 30
        
    def process_predictions(self, predictions: dict, video_frame: VideoFrame):
        """Xử lý predictions với optimizations"""
        
        self.frame_count += 1
        
        # Frame skip logic
        if FRAME_SKIP > 0 and self.frame_count % (FRAME_SKIP + 1) != 0:
            return
            
        self.process_count += 1
        
        # Calculate real-time FPS
        current_time = time.time()
        frame_time = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        instant_fps = 1.0 / frame_time if frame_time > 0 else 0
        self.fps_buffer.append(instant_fps)
        if len(self.fps_buffer) > self.fps_buffer_size:
            self.fps_buffer.pop(0)
        avg_fps = sum(self.fps_buffer) / len(self.fps_buffer)
        
        # Get frame
        frame = video_frame.image.copy()
        
        # Process predictions
        detections = []
        if predictions and "predictions" in predictions:
            for pred in predictions["predictions"]:
                confidence = pred["confidence"]
                class_name = pred["class"]
                
                if confidence >= CONFIDENCE_THRESHOLD:
                    detections.append({
                        'class_name': class_name,
                        'confidence': confidence,
                        'x': pred["x"],
                        'y': pred["y"],
                        'width': pred["width"],
                        'height': pred["height"]
                    })
        
        # Update tracker
        track_result = self.tracker.update(detections, self.frame_count)
        
        # Log detections
        if detections:
            self.logger.log_multiple(detections, self.frame_count)
        
        # ============ DRAWING ============
        
        # Draw bounding boxes
        for det in detections:
            x = int(det['x'] - det['width'] / 2)
            y = int(det['y'] - det['height'] / 2)
            w = int(det['width'])
            h = int(det['height'])
            
            # Green box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Label with background
            label = f"{det['class_name']}: {det['confidence']:.0%}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, y - label_h - 10), (x + label_w + 4, y), (0, 255, 0), -1)
            cv2.putText(frame, label, (x + 2, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Status bar background
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
        
        # FPS display
        fps_color = (0, 255, 0) if avg_fps >= 15 else (0, 255, 255) if avg_fps >= 10 else (0, 0, 255)
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)
        
        # Current signal display
        if track_result['current_signal']:
            signal_text = f"Signal: {track_result['current_signal']} ({track_result['confidence']:.0%})"
            cv2.putText(frame, signal_text, (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Sequence display
        if SHOW_SEQUENCE and self.tracker.sequence:
            seq_str = "Sequence: " + " ".join([s['signal'] for s in self.tracker.sequence[-8:]])
            cv2.putText(frame, seq_str, (10, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Frame count (right side)
        frame_text = f"Frame: {self.frame_count}"
        cv2.putText(frame, frame_text, (frame.shape[1] - 150, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Show frame
        cv2.imshow("Semaphore Detector - Optimized", frame)
        
        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self._shutdown()
        elif key == ord('r'):
            # Reset sequence
            self.tracker.reset()
            print("🔄 Sequence reset")
        elif key == ord('s'):
            # Print summary
            self.logger.print_summary()
            print(f"📝 Sequence: {self.tracker.get_sequence_string()}")
        
        # Periodic status
        if self.frame_count % 100 == 0:
            elapsed = time.time() - self.start_time
            overall_fps = self.frame_count / elapsed if elapsed > 0 else 0
            print(f"📊 Frame {self.frame_count} | FPS: {avg_fps:.1f} (avg: {overall_fps:.1f}) | "
                  f"Detections: {self.logger.total_detections}")
    
    def _shutdown(self):
        """Clean shutdown"""
        elapsed = time.time() - self.start_time
        avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        print("\n" + "="*60)
        print("🛑 SHUTTING DOWN")
        print("="*60)
        print(f"⏱️  Runtime: {elapsed:.1f}s")
        print(f"🎬 Total frames: {self.frame_count}")
        print(f"⚡ Average FPS: {avg_fps:.2f}")
        
        self.logger.print_summary()
        
        if self.tracker.sequence:
            print(f"\n📝 Final sequence: {self.tracker.get_sequence_string()}")
        
        print("="*60)
        pipeline.terminate()


# ============================================================
# 5. MAIN
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("🚀 SEMAPHORE DETECTOR - OPTIMIZED VERSION")
    print("="*60)
    print(f"📹 Camera: {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS}fps")
    print(f"🎯 Confidence threshold: {CONFIDENCE_THRESHOLD:.0%}")
    print(f"⏱️  Debounce time: {DEBOUNCE_TIME}s")
    print(f"📊 CSV logging: {'ON' if ENABLE_CSV_LOG else 'OFF'}")
    print("="*60)
    print("Controls: [Q] Quit | [R] Reset sequence | [S] Show summary")
    print("="*60)
    
    # Initialize processor
    processor = OptimizedVideoProcessor()
    
    # API credentials
    API_KEY = "ylFu6Gi5msSoDxbPC9Sl"
    MODEL_ID = "semaphore-dataset-1wlaa/1"
    
    # Initialize pipeline
    pipeline = InferencePipeline.init(
        api_key=API_KEY,
        model_id=MODEL_ID,
        video_reference=0,
        max_fps=CAMERA_FPS,
        on_prediction=processor.process_predictions,
    )
    
    # Configure camera (if possible)
    try:
        # Try to access underlying capture
        if hasattr(pipeline, '_video_source'):
            cap = pipeline._video_source
            if hasattr(cap, 'set'):
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER)
                print(f"✅ Camera configured: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    except Exception as e:
        print(f"⚠️  Could not configure camera directly: {e}")
    
    try:
        pipeline.start()
        pipeline.join()
    except KeyboardInterrupt:
        print("\n⛔ Interrupted by user")
    finally:
        cv2.destroyAllWindows()
        print("✅ Windows closed")
