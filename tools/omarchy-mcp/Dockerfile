FROM python:3.11-slim

WORKDIR /app

# Install git and other runtime deps (claude/codex would be added by user)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8911

ENV PORT=8911
ENV BIND=0.0.0.0

CMD ["python3", "server.py"]