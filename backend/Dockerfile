FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model file (assumes best_model.pth is in the same directory as Dockerfile)
COPY best_model.pth /app/model.pth

# Copy application code
COPY main.py .

# Expose port (Hugging Face Spaces uses 7860)
EXPOSE 7860

# Run the application on port 7860 for Hugging Face Spaces
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
