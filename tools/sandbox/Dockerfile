FROM python:3.11-slim

WORKDIR /

# Minimal deps for sandbox_server
RUN pip install --no-cache-dir aiohttp

# Copy sandbox server
COPY tools/sandbox_server.py /sandbox_server.py

EXPOSE 8080

CMD ["python", "/sandbox_server.py", "--port", "8080", "--slots", "10", "--data-dir", "/data"]
