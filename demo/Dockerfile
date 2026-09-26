# Live demo image for a Hugging Face Docker Space (free CPU hardware, port 7860).
# Built by demo/deploy_hf_space.py; locally:
#   docker build -f demo/Dockerfile -t infinity-demo .   &&   docker run -p 7860:7860 infinity-demo
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt requirements-dev.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-dev.txt

# Spaces run the container as uid 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app
COPY --chown=user . .

ENV YOLO_OFFLINE=1 DEMO_PROFILE=cpu DEMO_MAX_SECONDS=120 DEMO_MAX_MB=2500 DEMO_JOBS_DIR=/tmp/demo_jobs
EXPOSE 7860
CMD ["uvicorn", "demo.server:app", "--host", "0.0.0.0", "--port", "7860"]
