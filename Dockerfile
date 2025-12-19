FROM python:3.10-slim

WORKDIR /app

# Make /app importable
ENV PYTHONPATH=/app

# Install inference dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ONLY runtime code
COPY app ./app
COPY model_dir ./model_dir

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
