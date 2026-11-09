FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock
COPY studyflow studyflow
COPY resources resources
COPY evals evals
COPY examples examples
COPY app.py .
RUN pip install --no-deps . && useradd --create-home --uid 10001 student && mkdir /app/data && chown student /app/data
USER student
ENV DATA_DIR=/app/data
EXPOSE 8501
CMD ["python", "-m", "studyflow.serve", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
