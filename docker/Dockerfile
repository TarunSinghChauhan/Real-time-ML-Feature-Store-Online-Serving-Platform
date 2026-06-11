FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for some ML libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

EXPOSE 8000 8501

# Default command (can be overridden in docker-compose)
CMD ["uvicorn", "serving.main:app", "--host", "0.0.0.0", "--port", "8000"]
