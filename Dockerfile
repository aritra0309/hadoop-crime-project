# ============================================================
# India Crime Intelligence Platform — Standalone Docker Image
# Runs the full pipeline on Spark local mode (no HDFS required)
# ============================================================

FROM python:3.11.9-slim-bookworm

# Install Java (required by Spark)
RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-17-jre-headless curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

WORKDIR /app

# Copy requirements and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt pytest

# Copy project files
COPY . .

# Create local directories that mimic HDFS paths for local mode
RUN mkdir -p /tmp/crime/input /tmp/crime/output output/dashboard_data

# Copy data to local input directory
RUN cp data/*.csv /tmp/crime/input/

# Expose port for dashboard (optional)
EXPOSE 8080

# Default: run the full pipeline in local mode
CMD ["bash", "scripts/run_pipeline_local.sh"]
