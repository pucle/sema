"""
Semaphore Detector - FastAPI Backend
=====================================
Real-time AI detection server with WebSocket & HTTP endpoints
Designed for HuggingFace Spaces deployment
"""

import os
import time
import base64
import asyncio
from io import BytesIO
from datetime import datetime
from typing import Optional, Dict, List, Any
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# ============================================================
# CONFIGURATION
# ============================================================

# Roboflow credentials (from your existing code)
API_KEY = os.getenv("ROBOFLOW_API_KEY", "ylFu6Gi5msSoDxbPC9Sl")
MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID", "semaphore-dataset-1wlaa/1")

# Detection settings
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.35"))

# Device settings - prioritize GPU if available
DEVICE = os.getenv("INFERENCE_DEVICE", "auto")  # auto, cpu, cuda

# ============================================================
# MODEL LOADER
# ============================================================

model = None
model_loaded = False

def load_model():
    """Load Roboflow model with GPU preference"""
    global model, model_loaded
    
    print("🔄 Loading model...")
    
    # Configure environment for GPU if available
    if DEVICE == "auto":
        # Try to use GPU, fallback to CPU
        try:
            import torch
            if torch.cuda.is_available():
                os.environ["ROBOFLOW_INFERENCE_DEVICE"] = "cuda"
                print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                os.environ["ROBOFLOW_INFERENCE_DEVICE"] = "cpu"
                print("ℹ️ No GPU detected, using CPU")
        except ImportError:
            os.environ["ROBOFLOW_INFERENCE_DEVICE"] = "cpu"
            print("ℹ️ PyTorch not available, using CPU")
    else:
        os.environ["ROBOFLOW_INFERENCE_DEVICE"] = DEVICE
    
    # Optimize CPU threads
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    
    try:
        from inference import get_model
        model = get_model(model_id=MODEL_ID, api_key=API_KEY)
        model_loaded = True
        print(f"✅ Model loaded: {MODEL_ID}")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        model_loaded = False

# ============================================================
# APP SETUP
# ============================================================

app = FastAPI(
    title="Semaphore Detector API",
    description="Real-time semaphore signal detection API",
    version="1.0.0"
)

# CORS - Allow all origins for demo (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session storage for viewer mode
session_results: Dict[str, Dict[str, Any]] = defaultdict(dict)
active_connections: Dict[str, List[WebSocket]] = defaultdict(list)

# ============================================================
# STARTUP EVENT
# ============================================================

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

# ============================================================
# ROUTES
# ============================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "Semaphore Detector API",
        "model_loaded": model_loaded,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_id": MODEL_ID,
        "device": os.environ.get("ROBOFLOW_INFERENCE_DEVICE", "unknown"),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/process-frame")
async def process_frame(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None)
):
    """
    Process a single frame and return detections
    
    - **file**: JPEG image file
    - **session_id**: Optional session ID for viewer sharing
    """
    if not model_loaded:
        return JSONResponse(
            status_code=503,
            content={"error": "Model not loaded", "detections": []}
        )
    
    start_time = time.time()
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid image", "detections": []}
            )
        
        # Run inference in a separate thread to avoid blocking the event loop
        results = await asyncio.to_thread(model.infer, img)
        results = results[0]
        predictions = results.predictions
        
        # Format detections
        detections = []
        for pred in predictions:
            if hasattr(pred, 'class_name'):
                confidence = float(pred.confidence)
                if confidence >= CONFIDENCE_THRESHOLD:
                    detections.append({
                        "class": pred.class_name,
                        "confidence": confidence,
                        "x": float(pred.x),
                        "y": float(pred.y),
                        "width": float(pred.width),
                        "height": float(pred.height),
                        "bbox": [
                            float(pred.x - pred.width / 2),
                            float(pred.y - pred.height / 2),
                            float(pred.x + pred.width / 2),
                            float(pred.y + pred.height / 2)
                        ]
                    })
        
        # Store results for viewer mode
        result_data = {
            "detections": detections,
            "timestamp": time.time(),
            "latency": (time.time() - start_time) * 1000
        }
        
        if session_id:
            session_results[session_id] = result_data
            # Broadcast to viewers
            await broadcast_to_viewers(session_id, result_data)
        
        return result_data
        
    except Exception as e:
        print(f"❌ Processing error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "detections": []}
        )

@app.get("/api/latest-results")
async def get_latest_results(session: str = Query(...)):
    """Get latest results for a session (for viewer mode)"""
    if session in session_results:
        return session_results[session]
    return {"detections": [], "message": "No results yet"}

# ============================================================
# WEBSOCKET ENDPOINT
# ============================================================

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time streaming
    
    Client sends: base64 encoded JPEG image
    Server responds: JSON with detections
    """
    await websocket.accept()
    print(f"📡 WebSocket connection accepted")
    
    try:
        while True:
            # Receive base64 image
            print("DEBUG: Waiting for WebSocket message...")
            data = await websocket.receive_text()
            print(f"DEBUG: Received message, length: {len(data)}")
            
            if not model_loaded:
                await websocket.send_json({"error": "Model not loaded", "detections": []})
                continue
            
            start_time = time.time()
            
            try:
                # Parse base64 image
                if ',' in data:
                    # Data URL format: data:image/jpeg;base64,xxxxx
                    header, encoded = data.split(',', 1)
                else:
                    encoded = data
                
                img_data = base64.b64decode(encoded)
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is None:
                    await websocket.send_json({"error": "Invalid image", "detections": []})
                    continue
                
                # Run inference in a separate thread to avoid blocking the event loop
                print(f"DEBUG: Starting inference for WS frame...")
                results = await asyncio.to_thread(model.infer, img)
                results = results[0]
                predictions = results.predictions
                print(f"DEBUG: Inference complete. Detections count: {len(predictions)}")
                
                # Format detections
                detections = []
                for pred in predictions:
                    if hasattr(pred, 'class_name'):
                        confidence = float(pred.confidence)
                        if confidence >= CONFIDENCE_THRESHOLD:
                            detections.append({
                                "class": pred.class_name,
                                "confidence": confidence,
                                "x": float(pred.x),
                                "y": float(pred.y),
                                "width": float(pred.width),
                                "height": float(pred.height),
                                "bbox": [
                                    float(pred.x - pred.width / 2),
                                    float(pred.y - pred.height / 2),
                                    float(pred.x + pred.width / 2),
                                    float(pred.y + pred.height / 2)
                                ]
                            })
                
                latency = (time.time() - start_time) * 1000
                
                result_data = {
                    "detections": detections,
                    "timestamp": time.time(),
                    "latency": latency
                }
                
                await websocket.send_json(result_data)
                
            except Exception as e:
                print(f"⚠️ WebSocket processing error: {e}")
                await websocket.send_json({"error": str(e), "detections": []})
                
    except WebSocketDisconnect:
        print("📡 WebSocket client disconnected")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")

@app.websocket("/ws/viewer/{session_id}")
async def viewer_websocket(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for viewers to receive live results"""
    await websocket.accept()
    active_connections[session_id].append(websocket)
    
    try:
        # Send current results immediately
        if session_id in session_results:
            await websocket.send_json(session_results[session_id])
        
        # Keep connection alive
        while True:
            await websocket.receive_text()
            
    except WebSocketDisconnect:
        active_connections[session_id].remove(websocket)
    except Exception as e:
        print(f"Viewer WebSocket error: {e}")
        if websocket in active_connections[session_id]:
            active_connections[session_id].remove(websocket)

async def broadcast_to_viewers(session_id: str, data: dict):
    """Broadcast results to all viewers of a session"""
    if session_id in active_connections:
        for ws in active_connections[session_id]:
            try:
                await ws.send_json(data)
            except Exception:
                pass

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Semaphore Detector API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 SEMAPHORE DETECTOR API SERVER")
    print("=" * 60)
    print(f"📍 Host: {args.host}")
    print(f"📍 Port: {args.port}")
    print(f"📦 Model: {MODEL_ID}")
    print("=" * 60)
    
    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )
