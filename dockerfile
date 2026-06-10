# Stage 1: Build stage
FROM python:3.13-slim as builder

WORKDIR /app

# Install only necessary build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

# Install CPU-ONLY PyTorch (This saves ~4GB alone!)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime stage
FROM python:3.13-slim
WORKDIR /app

# Copy only the installed packages from the builder
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy your code
COPY . .

# CRITICAL: Ensure you aren't copying 10GB of models 
# (See .dockerignore below)

EXPOSE 3001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3001"]