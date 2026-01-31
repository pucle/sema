import requests
import cv2
import numpy as np
import time

# Configuration
BASE_URL = "http://localhost:8000"

def test_health():
    print(f"Testing Health Check at {BASE_URL}/health...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health Check Passed:", response.json())
            return True
        else:
            print("❌ Health Check Failed:", response.status_code)
            return False
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return False

def test_process_frame():
    print(f"\nTesting Frame Processing at {BASE_URL}/api/process-frame...")
    
    # Create a dummy image (black square)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw a simulated semaphore signal (just a white rectangle)
    cv2.rectangle(img, (200, 150), (400, 350), (255, 255, 255), -1)
    
    # Encode to JPEG
    _, img_encoded = cv2.imencode('.jpg', img)
    files = {'file': ('test.jpg', img_encoded.tobytes(), 'image/jpeg')}
    
    try:
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/api/process-frame", files=files)
        latency = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Processing Passed! Latency: {latency:.1f}ms")
            print("   Response Keys:", list(result.keys()))
            if 'detections' in result:
                print(f"   Detections: {len(result['detections'])}")
            return True
        else:
            print("❌ Processing Failed:", response.status_code, response.text)
            return False
    except Exception as e:
        print(f"❌ Process Error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Backend Verification...")
    
    health_ok = test_health()
    if health_ok:
        process_ok = test_process_frame()
        
    if health_ok and process_ok:
        print("\n✨ ALL TESTS PASSED! Backend is ready for deployment.")
    else:
        print("\n⚠️ SOME TESTS FAILED. Check server logs.")
