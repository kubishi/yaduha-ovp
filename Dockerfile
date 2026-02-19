FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "yaduha[api,agents] @ git+https://github.com/kubishi/yaduha-2.git"

COPY pyproject.toml ./
COPY yaduha_ovp/ ./yaduha_ovp/
RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "yaduha.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
