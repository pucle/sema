"""
SEMAPHORE DETECTOR - ULTRA OPTIMIZED VERSION (CPU)
===================================================
Phiên bản tối ưu dùng CPU:
- Threading riêng cho inference và display
- FPS mượt (~30fps) cho display dù inference chậm
- Real-time detection printing
- Signal tracking và smoothing
- Sử dụng CPU thay vì GPU yếu
"""

import cv2
import time
import os
import onnxruntime as ort
from datetime import datetime
from threading import Thread, Lock
import queue

# Import detection modules
from detection_logger import DetectionLogger, SignalTracker

# ============================================================
# CONFIGURATION  
# ============================================================

# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_BUFFER = 1

# Detection settings  
CONFIDENCE_THRESHOLD = 0.60
DEBOUNCE_TIME = 0.3
CONFIRM_FRAMES = 2

# Display settings
TARGET_DISPLAY_FPS = 30  # Display luôn mượt 30fps

# Logging
ENABLE_CSV_LOG = True  
LOG_FILE = "detections.csv"

# Roboflow credentials
API_KEY = "ylFu6Gi5msSoDxbPC9Sl"
MODEL_ID = "semaphore-dataset-1wlaa/1"


# ============================================================
# THREADED DETECTOR CLASS
# ============================================================

class ThreadedSemaphoreDetector:
    """
    Detector với threading để display mượt dù inference chậm.
    - Thread 1: Camera capture (liên tục)
    - Thread 2: Inference (chậm nhưng không block display)
    - Main: Display + UI
    """
    
    def __init__(self):
        # Frame buffers (protected by locks)
        self.current_frame = None
        self.display_frame = None
        self.frame_lock = Lock()
        
        # Detection results
        self.latest_detections = []
        self.detection_lock = Lock()
        
        # Stats
        self.frame_count = 0
        self.inference_count = 0
        self.start_time = time.time()
        self.inference_fps = 0
        self.display_fps = 0
        
        # Detection logging
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
        
        # Control flags
        self.running = False
        
        # Camera
        self.cap = None
        
        # Model (loaded later)
        self.model = None
        
    def _init_camera(self):
        """Initialize camera with optimized settings"""
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER)
        
        # Read actual settings
        actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"📹 Camera: {int(actual_w)}x{int(actual_h)}")
        
    def _init_model(self):
        """Load Roboflow model - Using CPU"""
        print("🔄 Loading model (CPU mode)...")
        
        # Force CPU - better than weak integrated GPU
        os.environ["ROBOFLOW_INFERENCE_DEVICE"] = "cpu"
        # Optimize CPU threads
        os.environ["OMP_NUM_THREADS"] = "4"
        os.environ["MKL_NUM_THREADS"] = "4"
        print("✅ CPU mode enabled (4 threads)")
        
        from inference import get_model
        self.model = get_model(model_id=MODEL_ID, api_key=API_KEY)
        print("✅ Model loaded")
        
    def _capture_thread(self):
        """Thread: Capture frames continuously"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.current_frame = frame.copy()
                    self.frame_count += 1
            time.sleep(1/60)  # 60fps capture
    
    def _inference_thread(self):
        """Thread: Run inference on latest frame"""
        last_inference_time = time.time()
        inference_times = []
        
        while self.running:
            # Get latest frame
            with self.frame_lock:
                if self.current_frame is None:
                    continue
                frame = self.current_frame.copy()
                frame_num = self.frame_count
            
            # Run inference
            start = time.time()
            try:
                results = self.model.infer(frame)[0]
                predictions = results.predictions
                
                # Convert to list of dicts
                detections = []
                for pred in predictions:
                    if hasattr(pred, 'class_name'):
                        det = {
                            'class_name': pred.class_name,
                            'confidence': pred.confidence,
                            'x': pred.x,
                            'y': pred.y, 
                            'width': pred.width,
                            'height': pred.height
                        }
                    else:
                        det = {
                            'class_name': pred.get('class', ''),
                            'confidence': pred.get('confidence', 0),
                            'x': pred.get('x', 0),
                            'y': pred.get('y', 0),
                            'width': pred.get('width', 0),
                            'height': pred.get('height', 0)
                        }
                    
                    if det['confidence'] >= CONFIDENCE_THRESHOLD:
                        detections.append(det)
                
                # Update results
                with self.detection_lock:
                    self.latest_detections = detections
                    self.inference_count += 1
                
                # Update tracker and logger
                self.tracker.update(detections, frame_num)
                if detections:
                    self.logger.log_multiple(detections, frame_num)
                    
            except Exception as e:
                print(f"⚠️ Inference error: {e}")
            
            # Calculate inference FPS
            inference_time = time.time() - start
            inference_times.append(inference_time)
            if len(inference_times) > 10:
                inference_times.pop(0)
            avg_time = sum(inference_times) / len(inference_times)
            self.inference_fps = 1.0 / avg_time if avg_time > 0 else 0
            
    def _draw_frame(self, frame):
        """Draw detections and UI on frame"""
        # Get latest detections
        with self.detection_lock:
            detections = self.latest_detections.copy()
        
        # Draw bounding boxes
        for det in detections:
            x = int(det['x'] - det['width'] / 2)
            y = int(det['y'] - det['height'] / 2)
            w = int(det['width'])
            h = int(det['height'])
            
            # Box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Label
            label = f"{det['class_name']}: {det['confidence']:.0%}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, y - lh - 10), (x + lw + 4, y), (0, 255, 0), -1)
            cv2.putText(frame, label, (x + 2, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Status bar
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 90), (30, 30, 30), -1)
        
        # Display FPS (always smooth)
        fps_color = (0, 255, 0) if self.display_fps >= 25 else (0, 255, 255)
        cv2.putText(frame, f"Display: {self.display_fps:.0f}fps", (10, 22),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)
        
        # Inference FPS
        inf_color = (0, 255, 0) if self.inference_fps >= 5 else (0, 165, 255)
        cv2.putText(frame, f"Inference: {self.inference_fps:.1f}fps", (10, 48),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, inf_color, 2)
        
        # Current signal
        if self.tracker.current_signal:
            sig = f"Signal: {self.tracker.current_signal} ({self.tracker.current_confidence:.0%})"
            cv2.putText(frame, sig, (200, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Sequence
        if self.tracker.sequence:
            seq = "Seq: " + " ".join([s['signal'] for s in self.tracker.sequence[-10:]])
            cv2.putText(frame, seq, (10, 78),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
        # Detection count
        det_text = f"Detections: {self.logger.total_detections}"
        cv2.putText(frame, det_text, (frame.shape[1] - 180, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
        return frame
    
    def run(self):
        """Main loop"""
        print("="*60)
        print("🚀 SEMAPHORE DETECTOR - ULTRA OPTIMIZED")
        print("="*60)
        print("Controls: [Q] Quit | [R] Reset | [S] Summary")
        print("="*60)
        
        # Initialize
        self._init_camera()
        self._init_model()
        
        self.running = True
        self.start_time = time.time()
        
        # Start threads
        capture_thread = Thread(target=self._capture_thread, daemon=True)
        inference_thread = Thread(target=self._inference_thread, daemon=True)
        
        capture_thread.start()
        inference_thread.start()
        
        print("▶️  Starting detection loop...")
        
        # Display loop (main thread - always smooth)
        display_times = []
        last_display = time.time()
        last_status = time.time()
        
        try:
            while self.running:
                # Get frame
                with self.frame_lock:
                    if self.current_frame is None:
                        continue
                    frame = self.current_frame.copy()
                
                # Draw UI
                frame = self._draw_frame(frame)
                
                # Calculate display FPS
                now = time.time()
                dt = now - last_display
                last_display = now
                display_times.append(1.0 / dt if dt > 0 else 0)
                if len(display_times) > 30:
                    display_times.pop(0)
                self.display_fps = sum(display_times) / len(display_times)
                
                # Show
                cv2.imshow("Semaphore Detector - Ultra", frame)
                
                # Keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.tracker.reset()
                    print("🔄 Sequence reset")
                elif key == ord('s'):
                    self.logger.print_summary()
                    print(f"📝 Sequence: {self.tracker.get_sequence_string()}")
                
                # Periodic status (every 30 seconds)
                if now - last_status > 30:
                    elapsed = now - self.start_time
                    print(f"📊 Status | Runtime: {elapsed:.0f}s | "
                          f"Display: {self.display_fps:.0f}fps | "
                          f"Inference: {self.inference_fps:.1f}fps | "
                          f"Detections: {self.logger.total_detections}")
                    last_status = now
                
                # Target display FPS
                time.sleep(max(0, 1/TARGET_DISPLAY_FPS - dt))
                
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            self._shutdown()
    
    def _shutdown(self):
        """Clean shutdown"""
        elapsed = time.time() - self.start_time
        
        print("\n" + "="*60)
        print("🛑 SHUTDOWN")
        print("="*60)
        print(f"⏱️  Runtime: {elapsed:.1f}s")
        print(f"🎬 Frames captured: {self.frame_count}")
        print(f"🔍 Inferences: {self.inference_count}")
        print(f"⚡ Avg inference FPS: {self.inference_fps:.2f}")
        
        self.logger.print_summary()
        
        if self.tracker.sequence:
            print(f"\n📝 Final sequence: {self.tracker.get_sequence_string()}")
        
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("="*60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    detector = ThreadedSemaphoreDetector()
    detector.run()
