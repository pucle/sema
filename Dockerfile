FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements from backend folder
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code from backend folder
COPY backend/app.py .

# Expose port (HF Spaces requirement)
EXPOSE 7860

# Run the application
# We use the same command as the original Dockerfile
# Force Rebuild: 2026-02-01 18:24
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
