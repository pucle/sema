---
title: Semaphore Detector
emoji: 🚦
colorFrom: indigo
colorTo: green
sdk: docker
pinned: false
---

# 🚦 Semaphore Detector - Webcam AI Streaming

Real-time semaphore signal detection using AI, deployed on free cloud infrastructure.

## 📁 Project Structure

```
sema/
├── frontend/           # Web application
│   ├── index.html      # Main page
│   ├── style.css       # Styling
│   ├── app.js          # Application logic
│   └── config.js       # Configuration
├── backend/            # Server application
│   ├── app.py          # FastAPI server
│   ├── requirements.txt# Dependencies
│   ├── Dockerfile      # For HuggingFace Spaces
│   └── README.md       # API documentation
├── colab_server.ipynb  # Google Colab (GPU) option
└── README.md           # This file
```

## 🚀 Quick Start

### Option A: Local Testing

1. **Start Backend**
   ```bash
   cd backend
   pip install -r requirements.txt
   python app.py
   ```

2. **Start Frontend**
   ```bash
   cd frontend
   python -m http.server 3000
   ```

3. Open `http://localhost:3000`

### Option B: HuggingFace Spaces (Recommended for Production)

1. Create new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
2. Choose "Docker" SDK
3. Upload files from `backend/` folder
4. Wait for build & deploy

### Option C: Google Colab (GPU)

1. Open `colab_server.ipynb` in Google Colab
2. Enable GPU runtime
3. Run all cells
4. Copy ngrok URL to frontend `config.js`

## ⚙️ Configuration

Edit `frontend/config.js`:

```javascript
const CONFIG = {
    // After deploying backend, update these URLs:
    BACKEND_URL: 'https://YOUR-SPACE.hf.space',
    WS_URL: 'wss://YOUR-SPACE.hf.space/ws/stream',
    
    FRAME_RATE: 10,      // FPS to send
    IMAGE_QUALITY: 0.7,  // JPEG quality
};
```

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/health` | GET | Detailed status |
| `/api/process-frame` | POST | Process single frame |
| `/api/latest-results` | GET | Get viewer results |
| `/ws/stream` | WebSocket | Real-time streaming |

## 📱 Features

- ✅ Real-time webcam processing
- ✅ Mobile-responsive UI
- ✅ Bounding box visualization
- ✅ Detection sequence tracking
- ✅ FPS & latency monitoring
- ✅ Viewer sharing mode
- ✅ WebSocket & HTTP fallback

## 🔒 Security Notes

- Change the Roboflow API key for production
- Restrict CORS origins in production
- Use environment variables for secrets

## 📄 License

MIT
