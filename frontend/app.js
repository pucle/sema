/**
 * Semaphore Detector - Main Application
 * Real-time AI detection with webcam streaming
 */

class SemaphoreDetector {
    constructor() {
        // DOM Elements
        this.videoElement = document.getElementById('videoElement');
        this.overlayCanvas = document.getElementById('overlayCanvas');
        this.videoOverlay = document.getElementById('videoOverlay');
        this.connectionStatus = document.getElementById('connectionStatus');

        // Buttons
        this.btnStartCamera = document.getElementById('btnStartCamera');
        this.btnStopCamera = document.getElementById('btnStopCamera');
        this.btnToggleStream = document.getElementById('btnToggleStream');
        this.btnClearLog = document.getElementById('btnClearLog');
        this.btnCopyLink = document.getElementById('btnCopyLink');

        // Stats
        this.fpsValue = document.getElementById('fpsValue');
        this.latencyValue = document.getElementById('latencyValue');
        this.detectionsValue = document.getElementById('detectionsValue');

        // Detection display
        this.signalLetter = document.getElementById('signalLetter');
        this.signalConfidence = document.getElementById('signalConfidence');
        this.sequenceDisplay = document.getElementById('sequenceDisplay');
        this.logContainer = document.getElementById('logContainer');
        this.viewerPanel = document.getElementById('viewerPanel');
        this.viewerLink = document.getElementById('viewerLink');

        // State
        this.stream = null;
        this.isStreaming = false;
        this.ws = null;
        this.frameInterval = null;
        this.reconnectAttempts = 0;

        // Stats tracking
        this.framesSent = 0;
        this.totalLatency = 0;
        this.lastFrameTime = 0;
        this.fpsHistory = [];
        this.totalDetections = 0;
        this.detectionSequence = [];

        // Canvas context
        this.ctx = this.overlayCanvas.getContext('2d');

        // Session ID for viewer sharing
        this.sessionId = this.generateSessionId();

        // Bind events
        this.bindEvents();
    }

    generateSessionId() {
        return 'sess_' + Math.random().toString(36).substr(2, 9);
    }

    bindEvents() {
        this.btnStartCamera.addEventListener('click', () => this.startCamera());
        this.btnStopCamera.addEventListener('click', () => this.stopCamera());
        this.btnToggleStream.addEventListener('click', () => this.toggleStreaming());
        this.btnClearLog.addEventListener('click', () => this.clearLog());
        this.btnCopyLink?.addEventListener('click', () => this.copyViewerLink());
    }

    // ==========================================
    // CAMERA FUNCTIONS
    // ==========================================

    async startCamera() {
        try {
            this.showToast('Requesting camera access...', 'info');

            // Request camera with constraints
            const constraints = {
                video: {
                    width: { ideal: CONFIG.MAX_WIDTH },
                    height: { ideal: CONFIG.MAX_HEIGHT },
                    facingMode: 'user' // Front camera on mobile
                },
                audio: false
            };

            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            this.videoElement.srcObject = this.stream;

            // Wait for video to load
            await new Promise((resolve) => {
                this.videoElement.onloadedmetadata = resolve;
            });

            // Setup canvas size
            this.overlayCanvas.width = this.videoElement.videoWidth;
            this.overlayCanvas.height = this.videoElement.videoHeight;

            // Update UI
            this.videoOverlay.classList.add('hidden');
            this.btnStartCamera.disabled = true;
            this.btnStopCamera.disabled = false;
            this.btnToggleStream.disabled = false;

            this.showToast('Camera started successfully!', 'success');
            this.log('📷 Camera started');

        } catch (error) {
            console.error('Camera error:', error);
            this.handleCameraError(error);
        }
    }

    handleCameraError(error) {
        let message = 'Camera error: ';

        if (error.name === 'NotAllowedError') {
            message += 'Permission denied. Please allow camera access.';
        } else if (error.name === 'NotFoundError') {
            message += 'No camera found on this device.';
        } else if (error.name === 'NotReadableError') {
            message += 'Camera is in use by another application.';
        } else {
            message += error.message;
        }

        this.showToast(message, 'error');
        this.log('❌ ' + message);
    }

    stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }

        this.stopStreaming();

        this.videoElement.srcObject = null;
        this.videoOverlay.classList.remove('hidden');
        this.btnStartCamera.disabled = false;
        this.btnStopCamera.disabled = true;
        this.btnToggleStream.disabled = true;

        // Clear canvas
        this.ctx.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);

        this.showToast('Camera stopped', 'info');
        this.log('📷 Camera stopped');
    }

    // ==========================================
    // STREAMING FUNCTIONS
    // ==========================================

    toggleStreaming() {
        if (this.isStreaming) {
            this.stopStreaming();
        } else {
            this.startStreaming();
        }
    }

    startStreaming() {
        this.isStreaming = true;
        this.btnToggleStream.innerHTML = '<span class="btn-icon">⏹️</span> Stop Streaming';
        this.btnToggleStream.classList.remove('btn-secondary');
        this.btnToggleStream.classList.add('btn-danger');

        this.updateConnectionStatus('processing', 'Connecting...');

        // Show viewer panel
        if (CONFIG.ENABLE_VIEWER_MODE) {
            this.setupViewerLink();
        }

        // Try WebSocket first, fallback to HTTP
        this.connectWebSocket();

        this.log('📡 Streaming started');
    }

    stopStreaming() {
        this.isStreaming = false;
        this.btnToggleStream.innerHTML = '<span class="btn-icon">📡</span> Start Streaming';
        this.btnToggleStream.classList.remove('btn-danger');
        this.btnToggleStream.classList.add('btn-secondary');

        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }

        if (this.frameInterval) {
            clearInterval(this.frameInterval);
            this.frameInterval = null;
        }

        this.updateConnectionStatus('disconnected', 'Disconnected');
        this.viewerPanel.style.display = 'none';

        this.log('📡 Streaming stopped');
    }

    connectWebSocket() {
        try {
            this.ws = new WebSocket(CONFIG.WS_URL);

            this.ws.onopen = () => {
                console.log('WebSocket connected');
                this.updateConnectionStatus('connected', 'Connected (WS)');
                this.reconnectAttempts = 0;
                this.startFrameCapture();
                this.showToast('Connected to server!', 'success');
            };

            this.ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleDetectionResults(data);
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                const errorMessage = error.message || 'WebSocket Connection Error';
                this.log(`⚠️ WebSocket error: ${errorMessage}, falling back to HTTP`);
                this.updateConnectionStatus('warning', 'WS Error, using HTTP');
                this.startHTTPPolling();
            };

            this.ws.onclose = (event) => {
                console.log('WebSocket closed:', event.code, event.reason);
                if (this.isStreaming) {
                    if (event.code !== 1000) {
                        this.log(`📡 WebSocket closed (Code: ${event.code}). Reason: ${event.reason || 'Unknown'}`);
                    }
                    this.attemptReconnect();
                }
            };

        } catch (error) {
            console.error('WebSocket connection setup failed:', error);
            this.log('❌ WebSocket setup failed: ' + error.message);
            this.startHTTPPolling();
        }
    }

    startHTTPPolling() {
        if (this.frameInterval && !this.ws) return; // Already polling or WS is fallbacking

        this.updateConnectionStatus('connected', 'Connected (HTTP)');
        this.log('📡 Starting HTTP polling mode');
        this.startFrameCapture(true);
    }

    startFrameCapture(useHTTP = false) {
        if (this.frameInterval) {
            clearInterval(this.frameInterval);
        }

        const interval = 1000 / CONFIG.FRAME_RATE;

        this.frameInterval = setInterval(() => {
            if (!this.isStreaming || !this.stream) return;

            this.captureAndSendFrame(useHTTP);
        }, interval);
    }

    async captureAndSendFrame(useHTTP = false) {
        const startTime = performance.now();

        try {
            // Capture frame from video
            const canvas = document.createElement('canvas');
            canvas.width = CONFIG.MAX_WIDTH;
            canvas.height = CONFIG.MAX_HEIGHT;
            const ctx = canvas.getContext('2d');

            // Draw video frame (flip horizontally to match display)
            ctx.translate(canvas.width, 0);
            ctx.scale(-1, 1);
            ctx.drawImage(this.videoElement, 0, 0, canvas.width, canvas.height);

            // Get base64 image
            const imageData = canvas.toDataURL('image/jpeg', CONFIG.IMAGE_QUALITY);

            // Update FPS tracking
            this.updateFPS();

            if (useHTTP || !this.ws || this.ws.readyState !== WebSocket.OPEN) {
                // HTTP request
                await this.sendFrameHTTP(imageData, startTime);
            } else {
                // WebSocket
                this.ws.send(imageData);
            }

            this.framesSent++;

        } catch (error) {
            console.error('Frame capture error:', error);
        }
    }

    async sendFrameHTTP(imageData, startTime) {
        try {
            // Convert base64 to blob
            const response = await fetch(imageData);
            const blob = await response.blob();

            const formData = new FormData();
            formData.append('file', blob, 'frame.jpg');
            formData.append('session_id', this.sessionId);

            const result = await fetch(`${CONFIG.BACKEND_URL}/api/process-frame`, {
                method: 'POST',
                body: formData,
                signal: AbortSignal.timeout(CONFIG.REQUEST_TIMEOUT)
            });

            if (result.ok) {
                const data = await result.json();
                const latency = performance.now() - startTime;
                this.handleDetectionResults({ ...data, latency });
            }

        } catch (error) {
            if (error.name !== 'AbortError') {
                console.error('HTTP request error:', error);
            }
        }
    }

    attemptReconnect() {
        if (this.reconnectAttempts >= CONFIG.MAX_RECONNECT_ATTEMPTS) {
            this.updateConnectionStatus('disconnected', 'Connection failed');
            this.showToast('Failed to connect. Please check server status.', 'error');
            this.stopStreaming();
            return;
        }

        this.reconnectAttempts++;
        this.updateConnectionStatus('processing', `Reconnecting (${this.reconnectAttempts}/${CONFIG.MAX_RECONNECT_ATTEMPTS})...`);

        setTimeout(() => {
            if (this.isStreaming) {
                this.connectWebSocket();
            }
        }, CONFIG.RECONNECT_DELAY);
    }

    // ==========================================
    // DETECTION HANDLING
    // ==========================================

    handleDetectionResults(data) {
        const latency = data.latency || 0;
        this.updateLatency(latency);

        // Clear previous drawings
        this.ctx.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);

        if (data.detections && data.detections.length > 0) {
            // Draw bounding boxes
            if (CONFIG.SHOW_BOUNDING_BOXES) {
                this.drawDetections(data.detections);
            }

            // Update current detection
            const topDetection = data.detections.reduce((prev, current) =>
                (prev.confidence > current.confidence) ? prev : current
            );

            this.updateCurrentDetection(topDetection);
            this.addToSequence(topDetection);
            this.totalDetections++;

            // Log
            this.log(`🎯 ${topDetection.class} (${(topDetection.confidence * 100).toFixed(1)}%)`);
        }

        // Update stats
        this.detectionsValue.textContent = this.totalDetections;
    }

    drawDetections(detections) {
        detections.forEach(det => {
            // Calculate box coordinates
            let x, y, w, h;

            if (det.bbox) {
                // Format: [x1, y1, x2, y2]
                x = det.bbox[0];
                y = det.bbox[1];
                w = det.bbox[2] - det.bbox[0];
                h = det.bbox[3] - det.bbox[1];
            } else if (det.x !== undefined) {
                // Format: center x, y, width, height
                x = det.x - det.width / 2;
                y = det.y - det.height / 2;
                w = det.width;
                h = det.height;
            } else {
                return;
            }

            // Scale to canvas size
            const scaleX = this.overlayCanvas.width / CONFIG.MAX_WIDTH;
            const scaleY = this.overlayCanvas.height / CONFIG.MAX_HEIGHT;

            x *= scaleX;
            y *= scaleY;
            w *= scaleX;
            h *= scaleY;

            // Draw box
            this.ctx.strokeStyle = CONFIG.BOX_COLOR;
            this.ctx.lineWidth = CONFIG.BOX_LINE_WIDTH;
            this.ctx.strokeRect(x, y, w, h);

            // Draw label background
            const label = `${det.class || det.class_name}: ${(det.confidence * 100).toFixed(0)}%`;
            this.ctx.font = CONFIG.LABEL_FONT;
            const textWidth = this.ctx.measureText(label).width;

            this.ctx.fillStyle = CONFIG.BOX_COLOR;
            this.ctx.fillRect(x, y - 25, textWidth + 10, 25);

            // Draw label text
            this.ctx.fillStyle = '#000';
            this.ctx.fillText(label, x + 5, y - 7);
        });
    }

    updateCurrentDetection(detection) {
        const className = detection.class || detection.class_name || '-';
        // Extract letter if it's a semaphore signal (e.g., "A", "B", etc.)
        const letter = className.toUpperCase().charAt(0);

        this.signalLetter.textContent = letter;
        this.signalConfidence.textContent = `${(detection.confidence * 100).toFixed(1)}%`;

        // Add pulse animation
        this.signalLetter.classList.add('pulse');
        setTimeout(() => this.signalLetter.classList.remove('pulse'), 300);
    }

    addToSequence(detection) {
        const className = detection.class || detection.class_name;
        const letter = className.toUpperCase().charAt(0);

        // Avoid duplicates
        if (this.detectionSequence.length === 0 ||
            this.detectionSequence[this.detectionSequence.length - 1] !== letter) {
            this.detectionSequence.push(letter);

            // Keep last 20
            if (this.detectionSequence.length > 20) {
                this.detectionSequence.shift();
            }

            this.updateSequenceDisplay();
        }
    }

    updateSequenceDisplay() {
        if (this.detectionSequence.length === 0) {
            this.sequenceDisplay.innerHTML = '<span class="placeholder-text">Waiting for detections...</span>';
            return;
        }

        this.sequenceDisplay.innerHTML = this.detectionSequence
            .map(letter => `<span class="signal-badge">${letter}</span>`)
            .join('');
    }

    // ==========================================
    // STATS & UI UPDATES
    // ==========================================

    updateFPS() {
        const now = performance.now();
        if (this.lastFrameTime > 0) {
            const fps = 1000 / (now - this.lastFrameTime);
            this.fpsHistory.push(fps);
            if (this.fpsHistory.length > 10) {
                this.fpsHistory.shift();
            }
            const avgFPS = this.fpsHistory.reduce((a, b) => a + b, 0) / this.fpsHistory.length;
            this.fpsValue.textContent = avgFPS.toFixed(1);
        }
        this.lastFrameTime = now;
    }

    updateLatency(latency) {
        this.latencyValue.textContent = `${latency.toFixed(0)} ms`;

        // Color code based on latency
        if (latency < 200) {
            this.latencyValue.style.color = 'var(--success)';
        } else if (latency < 500) {
            this.latencyValue.style.color = 'var(--warning)';
        } else {
            this.latencyValue.style.color = 'var(--danger)';
        }
    }

    updateConnectionStatus(status, text) {
        this.connectionStatus.className = 'connection-status ' + status;
        this.connectionStatus.querySelector('.status-text').textContent = text;
    }

    // ==========================================
    // LOGGING
    // ==========================================

    log(message) {
        const time = new Date().toLocaleTimeString();
        const entry = document.createElement('div');
        entry.className = 'log-entry';
        entry.innerHTML = `<span class="log-time">${time}</span> ${message}`;

        // Remove placeholder if exists
        const placeholder = this.logContainer.querySelector('.log-placeholder');
        if (placeholder) placeholder.remove();

        this.logContainer.insertBefore(entry, this.logContainer.firstChild);

        // Keep only last 50 entries
        while (this.logContainer.children.length > 50) {
            this.logContainer.removeChild(this.logContainer.lastChild);
        }
    }

    clearLog() {
        this.logContainer.innerHTML = '<p class="log-placeholder">Results will appear here...</p>';
        this.totalDetections = 0;
        this.detectionsValue.textContent = '0';
        this.detectionSequence = [];
        this.updateSequenceDisplay();
        this.signalLetter.textContent = '-';
        this.signalConfidence.textContent = '0%';
    }

    // ==========================================
    // VIEWER SHARING
    // ==========================================

    setupViewerLink() {
        const baseUrl = window.location.origin + window.location.pathname;
        const viewerUrl = `${baseUrl}?viewer=true&session=${this.sessionId}`;
        this.viewerLink.value = viewerUrl;
        this.viewerPanel.style.display = 'block';
    }

    copyViewerLink() {
        this.viewerLink.select();
        navigator.clipboard.writeText(this.viewerLink.value);
        this.showToast('Link copied to clipboard!', 'success');
    }

    // ==========================================
    // UTILITIES
    // ==========================================

    showToast(message, type = 'info') {
        const container = document.getElementById('toastContainer');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;

        container.appendChild(toast);

        setTimeout(() => {
            toast.remove();
        }, 4000);
    }
}

// ==========================================
// VIEWER MODE
// ==========================================

class ViewerMode {
    constructor(sessionId) {
        this.sessionId = sessionId;
        this.pollInterval = null;

        this.init();
    }

    init() {
        document.querySelector('.controls').style.display = 'none';
        document.querySelector('.video-section').innerHTML = `
            <div class="viewer-notice">
                <h2>📺 Viewer Mode</h2>
                <p>Watching live detection results</p>
                <p class="session-id">Session: ${this.sessionId}</p>
            </div>
        `;

        this.startPolling();
    }

    startPolling() {
        this.pollInterval = setInterval(() => this.fetchResults(), CONFIG.VIEWER_POLL_INTERVAL);
    }

    async fetchResults() {
        try {
            const response = await fetch(`${CONFIG.BACKEND_URL}/api/latest-results?session=${this.sessionId}`);
            if (response.ok) {
                const data = await response.json();
                this.displayResults(data);
            }
        } catch (error) {
            console.error('Viewer poll error:', error);
        }
    }

    displayResults(data) {
        if (data.detections && data.detections.length > 0) {
            const topDetection = data.detections[0];
            document.getElementById('signalLetter').textContent = topDetection.class?.charAt(0) || '-';
            document.getElementById('signalConfidence').textContent =
                `${(topDetection.confidence * 100).toFixed(1)}%`;
        }
    }
}

// ==========================================
// INITIALIZATION
// ==========================================

document.addEventListener('DOMContentLoaded', () => {
    // Check if viewer mode
    const urlParams = new URLSearchParams(window.location.search);
    const isViewer = urlParams.get('viewer') === 'true';
    const sessionId = urlParams.get('session');

    if (isViewer && sessionId) {
        new ViewerMode(sessionId);
    } else {
        window.app = new SemaphoreDetector();
    }
});
