# Dockerfile - Enterprise Invoice Payment Intelligence App
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8501

# Install system dependencies including Java for Spark
RUN apt-get update && apt-get install -y \
    default-jdk-headless \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set JAVA_HOME for Spark
ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Install core packages explicitly to ensure they're available
RUN pip install --no-cache-dir \
    streamlit==1.28.0 \
    pandas==2.0.0 \
    numpy==1.24.0 \
    plotly==5.15.0 \
    scikit-learn==1.3.0 \
    joblib==1.3.0

# Install enterprise packages
RUN pip install --no-cache-dir \
    pyspark==3.4.0 \
    tensorflow-cpu==2.12.0 \
    sqlalchemy==2.0.0 \
    psycopg2-binary==2.9.0 \
    redis==4.5.0

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs data/models/saved_models

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8501', timeout=10)"

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]