---
title: Semaphore Detector
emoji: 🚦
colorFrom: indigo
colorTo: green
sdk: docker
pinned: false
license: mit
---

# 🚦 Semaphore Signal Detector

Real-time semaphore signal detection using AI.

## Features

- **Real-time Detection**: Process webcam frames and detect semaphore signals
- **WebSocket Support**: Low-latency streaming via WebSocket
- **HTTP API**: REST endpoint for frame processing
- **Viewer Mode**: Share live detection results with others

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/health` | GET | Detailed health status |
| `/api/process-frame` | POST | Process a single frame |
| `/api/latest-results` | GET | Get latest results (viewer mode) |
| `/ws/stream` | WebSocket | Real-time streaming |
| `/ws/viewer/{session_id}` | WebSocket | Viewer mode connection |

## Usage

### HTTP Request
```bash
curl -X POST "https://YOUR-SPACE.hf.space/api/process-frame" \
  -F "file=@frame.jpg"
```

### Response Format
```json
{
  "detections": [
    {
      "class": "A",
      "confidence": 0.95,
      "x": 320,
      "y": 240,
      "width": 100,
      "height": 150,
      "bbox": [270, 165, 370, 315]
    }
  ],
  "timestamp": 1706712345.123,
  "latency": 45.2
}
```

## Model

This space uses the Semaphore Dataset model from Roboflow:
- Model ID: `semaphore-dataset-1wlaa/1`
- Type: Object Detection

## License

MIT
