FROM python:3.11-slim

WORKDIR /app

# Set PYTHONPATH so 'src' module imports resolve cleanly inside container
ENV PYTHONPATH=/app

# Install minimal C-libraries for OpenCV / Scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install lightweight CPU PyTorch wheel
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining production requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
