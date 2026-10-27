FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --create-home agent
COPY . /app
RUN python -m pip install --no-cache-dir '.[server,ml,gpu]' \
    && mkdir -p /workspace/outputs /cache && chown -R agent:agent /workspace /cache
ENV HF_HOME=/cache PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
USER agent
EXPOSE 8787
CMD ["python", "-m", "exec_feedback", "serve", "--config", "/workspace/local.json", "--host", "0.0.0.0"]
