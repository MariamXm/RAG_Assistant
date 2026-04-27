FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# system dependencies (important for torch + chroma)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# install dependencies first (better caching)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# copy project files
COPY . .

# expose ports
EXPOSE 8000
EXPOSE 8501

# run both services
CMD bash -c "uvicorn api:app --host 0.0.0.0 --port 8000 & streamlit run app.py --server.port 8501 --server.address 0.0.0.0"