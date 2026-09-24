# Evaluation image: same two commands as the task, inside the container.
#   docker build -t infinity .
#   docker run --gpus all -v /data/test:/data/test infinity \
#       python run_submission.py --videos /data/test --out predictions.json
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV YOLO_OFFLINE=1
