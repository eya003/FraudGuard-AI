FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PORT=8501
WORKDIR /app
COPY requirements.txt requirements-optional.txt ./
ARG INCLUDE_OPTIONAL=true
RUN pip install --no-cache-dir -r requirements.txt && \
    if [ "$INCLUDE_OPTIONAL" = "true" ]; then pip install --no-cache-dir -r requirements-optional.txt; fi
COPY . .
RUN mkdir -p data/uploads data/experiments && chmod +x deploy/start.sh
EXPOSE 8501
HEALTHCHECK --interval=35s --timeout=10s --start-period=80s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health',timeout=6)" || exit 1
CMD ["sh", "deploy/start.sh"]
