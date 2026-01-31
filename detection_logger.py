"""
Detection Logger - In và ghi log các tín hiệu semaphore được phát hiện
"""

import time
from datetime import datetime
from collections import deque
import csv
import os
from threading import Thread, Lock
import queue


class DetectionLogger:
    """
    Ghi nhận và in kết quả phát hiện với:
    - Debouncing để tránh spam
    - Lưu log vào CSV
    - Non-blocking print
    """
    
    def __init__(self, 
                 debounce_time: float = 0.5,
                 min_confidence: float = 0.7,
                 log_file: str = None,
                 print_enabled: bool = True):
        """
        Args:
            debounce_time: Thời gian tối thiểu giữa 2 lần in cùng 1 class (giây)
            min_confidence: Ngưỡng confidence tối thiểu để log
            log_file: Đường dẫn file CSV để lưu log (None = không lưu)
            print_enabled: Bật/tắt print ra console
        """
        self.debounce_time = debounce_time
        self.min_confidence = min_confidence
        self.print_enabled = print_enabled
        
        # Tracking last detection time per class
        self.last_detection_time = {}
        
        # Detection history
        self.history = deque(maxlen=1000)
        
        # CSV logging
        self.log_file = log_file
        self._init_csv()
        
        # Async print queue
        self.print_queue = queue.Queue()
        self.print_thread = Thread(target=self._print_worker, daemon=True)
        self.print_thread.start()
        
        # Stats
        self.total_detections = 0
        self.unique_signals = set()
        self.lock = Lock()
    
    def _init_csv(self):
        """Khởi tạo file CSV nếu được chỉ định"""
        if self.log_file:
            if not os.path.exists(self.log_file):
                with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['timestamp', 'class_name', 'confidence', 'frame_number'])
    
    def _print_worker(self):
        """Worker thread để print non-blocking"""
        while True:
            try:
                msg = self.print_queue.get(timeout=1)
                if msg:
                    print(msg)
            except queue.Empty:
                continue
    
    def log_detection(self, class_name: str, confidence: float, frame_number: int) -> bool:
        """
        Log một detection. Return True nếu được print (passed debounce).
        
        Args:
            class_name: Tên class được detect
            confidence: Độ tin cậy (0-1)
            frame_number: Số frame hiện tại
            
        Returns:
            True nếu detection được print, False nếu bị debounce
        """
        if confidence < self.min_confidence:
            return False
        
        current_time = time.time()
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # Check debounce
        should_print = False
        with self.lock:
            last_time = self.last_detection_time.get(class_name, 0)
            if current_time - last_time >= self.debounce_time:
                should_print = True
                self.last_detection_time[class_name] = current_time
                self.total_detections += 1
                self.unique_signals.add(class_name)
        
        # Add to history
        record = {
            'timestamp': timestamp,
            'class_name': class_name,
            'confidence': confidence,
            'frame_number': frame_number
        }
        self.history.append(record)
        
        # Write to CSV
        if self.log_file:
            try:
                with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp, class_name, f"{confidence:.4f}", frame_number])
            except Exception:
                pass
        
        # Print if passed debounce
        if should_print and self.print_enabled:
            msg = f"[{timestamp}] ✅ Detected: {class_name} ({confidence:.1%}) - Frame #{frame_number}"
            self.print_queue.put(msg)
        
        return should_print
    
    def log_multiple(self, detections: list, frame_number: int) -> list:
        """
        Log nhiều detections cùng lúc.
        
        Args:
            detections: List của dict có keys: class_name, confidence
            frame_number: Số frame hiện tại
            
        Returns:
            List các class_name được print
        """
        printed = []
        for det in detections:
            class_name = det.get('class_name') or det.get('class')
            confidence = det.get('confidence', 0)
            if self.log_detection(class_name, confidence, frame_number):
                printed.append(class_name)
        return printed
    
    def get_stats(self) -> dict:
        """Lấy thống kê"""
        return {
            'total_detections': self.total_detections,
            'unique_signals': len(self.unique_signals),
            'signals_seen': list(self.unique_signals),
            'history_size': len(self.history)
        }
    
    def get_recent(self, n: int = 10) -> list:
        """Lấy n detection gần nhất"""
        return list(self.history)[-n:]
    
    def print_summary(self):
        """In tổng kết"""
        stats = self.get_stats()
        print("\n" + "="*50)
        print("📊 DETECTION SUMMARY")
        print("="*50)
        print(f"Total detections logged: {stats['total_detections']}")
        print(f"Unique signals seen: {stats['unique_signals']}")
        if stats['signals_seen']:
            print(f"Signals: {', '.join(sorted(stats['signals_seen']))}")
        print("="*50)


class SignalTracker:
    """
    Theo dõi và xác nhận tín hiệu qua nhiều frame.
    Loại bỏ nhiễu bằng cách yêu cầu confirmation.
    """
    
    def __init__(self, 
                 confirm_frames: int = 3,
                 memory_frames: int = 10,
                 min_confidence: float = 0.6):
        """
        Args:
            confirm_frames: Số frame cần để confirm 1 signal
            memory_frames: Số frame lưu trong bộ nhớ
            min_confidence: Ngưỡng confidence tối thiểu
        """
        self.confirm_frames = confirm_frames
        self.memory_frames = memory_frames
        self.min_confidence = min_confidence
        
        # Voting buffer: class_name -> list of (frame_num, confidence)
        self.votes = {}
        
        # Current confirmed signal
        self.current_signal = None
        self.current_confidence = 0.0
        
        # Signal sequence
        self.sequence = []
        
    def update(self, detections: list, frame_number: int) -> dict:
        """
        Cập nhật với detections mới.
        
        Args:
            detections: List của detection dicts
            frame_number: Frame number hiện tại
            
        Returns:
            Dict với current_signal, is_new, sequence
        """
        # Filter by confidence
        valid = [d for d in detections if d.get('confidence', 0) >= self.min_confidence]
        
        # Get best detection
        if valid:
            best = max(valid, key=lambda x: x.get('confidence', 0))
            class_name = best.get('class_name') or best.get('class')
            confidence = best.get('confidence', 0)
            
            # Add vote
            if class_name not in self.votes:
                self.votes[class_name] = []
            self.votes[class_name].append((frame_number, confidence))
        
        # Clean old votes
        for cls in list(self.votes.keys()):
            self.votes[cls] = [(f, c) for f, c in self.votes[cls] 
                              if frame_number - f <= self.memory_frames]
            if not self.votes[cls]:
                del self.votes[cls]
        
        # Check for confirmed signal
        is_new = False
        for cls, vote_list in self.votes.items():
            if len(vote_list) >= self.confirm_frames:
                avg_conf = sum(c for _, c in vote_list) / len(vote_list)
                if cls != self.current_signal:
                    is_new = True
                    self.sequence.append({
                        'signal': cls,
                        'confidence': avg_conf,
                        'frame': frame_number
                    })
                self.current_signal = cls
                self.current_confidence = avg_conf
                break
        
        return {
            'current_signal': self.current_signal,
            'confidence': self.current_confidence,
            'is_new': is_new,
            'sequence': self.sequence[-10:]  # Last 10 signals
        }
    
    def get_sequence_string(self) -> str:
        """Lấy chuỗi tín hiệu đã phát hiện"""
        return ' '.join([s['signal'] for s in self.sequence])
    
    def reset(self):
        """Reset tracker"""
        self.votes = {}
        self.current_signal = None
        self.current_confidence = 0.0
        self.sequence = []
