FROM python:3.10

WORKDIR /app

# upgrade pip
RUN pip install --upgrade pip

# dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# code
COPY . .

# port FastAPI
EXPOSE 8000

# lancer API
CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000"]