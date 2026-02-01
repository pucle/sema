/**
 * Semaphore Detector - Configuration
 * Update these values after deploying backend
 */

const CONFIG = {
    // Backend URLs - Updated to HuggingFace Space
    BACKEND_URL: 'https://pucle-sema.hf.space',
    WS_URL: 'wss://pucle-sema.hf.space/ws/stream',

    // For HuggingFace Spaces:
    // BACKEND_URL: 'https://YOUR_USERNAME-semaphore-detector.hf.space',
    // WS_URL: 'wss://YOUR_USERNAME-semaphore-detector.hf.space/ws/stream',

    // Frame capture settings
    FRAME_RATE: 8,           // FPS to send to server (lower = better stability on CPU)
    IMAGE_QUALITY: 0.7,       // JPEG quality (0.1 - 1.0)
    MAX_WIDTH: 416,           // Reduced for faster CPU inference
    MAX_HEIGHT: 312,

    // Connection settings
    RECONNECT_DELAY: 2000,    // ms before reconnecting
    MAX_RECONNECT_ATTEMPTS: 5,
    REQUEST_TIMEOUT: 10000,   // ms timeout for HTTP requests

    // UI settings
    SHOW_BOUNDING_BOXES: true,
    BOX_COLOR: '#10b981',     // Green
    BOX_LINE_WIDTH: 3,
    LABEL_FONT: '16px Inter, sans-serif',

    // Viewer settings (for live sharing)
    ENABLE_VIEWER_MODE: true,
    VIEWER_POLL_INTERVAL: 500, // ms for viewer to poll results
};

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CONFIG;
}
