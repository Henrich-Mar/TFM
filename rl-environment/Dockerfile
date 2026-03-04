FROM python:3.11-slim

WORKDIR /app

# Install system dependencies and Rust
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    patchelf \
    && rm -rf /var/lib/apt/lists/* \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

ENV PATH="/root/.cargo/bin:${PATH}"

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Build and install Rust extension.
RUN maturin build --release --skip-auditwheel --interpreter python3.11
RUN pip install --no-cache-dir target/wheels/*.whl
RUN python -c "import rust_tfm_rl; print(rust_tfm_rl.backend_info())"

# Create necessary directories
RUN mkdir -p /app/models /app/logs /app/metrics /app/api

# Expose ports
EXPOSE 5000 6006

# Run the RL coordinator
CMD ["python", "coordinator.py"]
