# Live demo image (e.g. a Hugging Face Docker Space, which expects port 7860).
#   docker build -f demo/Dockerfile -t infinity-demo .
#   docker run -p 7860:7860 infinity-demo
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt
COPY . .
ENV YOLO_OFFLINE=1
EXPOSE 7860
CMD ["uvicorn", "demo.server:app", "--host", "0.0.0.0", "--port", "7860"]
