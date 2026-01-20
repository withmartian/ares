FROM python:3.12-slim

WORKDIR /app

# Install FastAPI and uvicorn
RUN pip install --no-cache-dir fastapi==0.115.6 uvicorn==0.34.0

# Copy the proxy application
COPY proxy.py /app/proxy.py

# Expose port for HTTP API
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "proxy:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
