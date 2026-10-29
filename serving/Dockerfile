# Build stage where we install all the packages in the python environment
FROM python:3.11-slim-bookworm AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Get git 
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install python venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install required python packages
COPY serving/requirements.txt .

RUN pip install --upgrade --no-cache-dir pip && \
    pip install --no-deps --no-cache-dir -r requirements.txt

# Move the configs and the functions
COPY configs/ ./configs/
COPY mma_functions/ ./mma_functions/
COPY serving/ ./serving/
# Copy the model to be deployed into the image
COPY models/xgb_forest_v4.4.0.joblib ./models/xgb_forest_v4.4.0.joblib

#############################################################
# Final image

FROM python:3.11-slim-bookworm

WORKDIR /app

# Copy the files and environment from the builder
COPY --from=builder /app .
COPY --from=builder /opt/venv /opt/venv

# Point python to the venv
ENV PATH="/opt/venv/bin:$PATH"

# Set the model path within the image
ENV ENSEMBLE_PATH="/app/models/xgb_forest_v4.4.0.joblib"

# Set the python path so that functions in mma_functions will be sourced correctly
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV OMP_NUM_THREADS=1

# Expose the port the service runs on
EXPOSE 8000

# Change user to non root
RUN useradd --uid 1000 --no-create-home mma_user
USER 1000

CMD ["uvicorn", "serving.main:app", "--host", "0.0.0.0", "--port", "8000"]



