FROM python:3.14-slim AS backend
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 ROAD_DATA_DIR=/app/data
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements-vision.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-vision.txt
COPY common ./common
COPY services ./services
COPY agent ./agent
RUN useradd --create-home --uid 10001 road && mkdir -p /app/data /app/.cache/tmp && chown -R road:road /app
USER road
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "services.gateway.app:app", "--host", "0.0.0.0", "--port", "8000"]

FROM node:24-alpine AS frontend-build
WORKDIR /web
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM nginx:alpine AS frontend
COPY scripts/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=frontend-build /web/dist /usr/share/nginx/html
EXPOSE 80
